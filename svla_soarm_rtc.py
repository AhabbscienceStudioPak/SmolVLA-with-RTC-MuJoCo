import torch
import gymnasium as gym
import gym_soarm
import numpy as np
import imageio
import cv2
from PIL import Image
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import threading
import time
from gym_soarm.constants import JOINTS
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.datasets.utils import load_json, cast_stats_to_numpy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
import json
from lerobot.configs.policies import PreTrainedConfig


class SmolVLAController:
    """
    Controller for running SmolVLA policy with RTC (Real-Time Chunking) 
    on SO-ARM robotic environment.
    
    This class handles:
    - Loading and configuring the SmolVLA policy with RTC
    - Managing the action queue for temporal consistency
    - Running the control loop with visualization
    - Recording video of the execution
    """
    
    def __init__(
        self,
        model_name: str = "lerobot/smolvla_base",
        stats_path: str = "svla_so101_pickplace.json",
        env_name: str = 'gym_soarm/PickAndPlaceCube-v0',
        task_instruction: str = "blue brick",
        execution_horizon: int = 8,
        max_guidance_weight: float = 5.0,
        prefix_attention_schedule: str = "EXP",
        inference_delay: int = 10,
        max_steps: int = 200,
        record_video: bool = True,
        device: str = None
    ):
        """
        Initialize the SmolVLA Controller.
        
        Args:
            model_name (str): HuggingFace model identifier for SmolVLA policy.
                Default: "lerobot/smolvla_base"
            
            stats_path (str): Path to JSON file containing dataset statistics for 
                normalization/denormalization of actions and observations.
                Default: "svla_so101_pickplace.json"
            
            env_name (str): Gymnasium environment ID for the SO-ARM simulator.
                Default: 'gym_soarm/PickAndPlaceCube-v0'
            
            task_instruction (str): Natural language instruction describing the task 
                for the vision-language-action model to perform.
                Default: "blue brick"
            
            execution_horizon (int): Number of future steps to blend with previous 
                chunk for temporal consistency in RTC. Higher values = smoother but 
                less reactive. Range: 1-20.
                Default: 8
            
            max_guidance_weight (float): Maximum weight for RTC guidance during 
                denoising. Higher values enforce stronger temporal consistency.
                Range: 1.0-10.0.
                Default: 5.0
            
            prefix_attention_schedule (str): Attention schedule strategy for RTC prefix.
                Options: "LINEAR", "EXP" (exponential), "CONSTANT"
                Default: "EXP"
            
            inference_delay (int): Number of steps between replanning. Lower values 
                = more frequent replanning (more reactive, more compute). Higher values 
                = less frequent replanning (smoother, less compute).
                Typical range: 5-20.
                Default: 10
            
            max_steps (int): Maximum number of environment steps to execute before 
                terminating the episode.
                Default: 200
            
            record_video (bool): Whether to record video frames for saving as MP4.
                Default: True
            
            device (str): Torch device to run inference on. Options: "cuda", "cpu", 
                or None for auto-detection.
                Default: None (auto-detect)
        """
        # Store configuration
        self.model_name = model_name
        self.stats_path = stats_path
        self.env_name = env_name
        self.task_instruction = task_instruction
        self.inference_delay = inference_delay
        self.max_steps = max_steps
        self.record_video = record_video
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Configure RTC
        self.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=execution_horizon,
            max_guidance_weight=max_guidance_weight,
            prefix_attention_schedule=getattr(RTCAttentionSchedule, prefix_attention_schedule)
        )
        
        # Load policy configuration and model
        self._load_policy()
        
        # Setup environment
        self._setup_environment()
        
        # Load statistics and create processors
        self._setup_processors()
        
        # Initialize action queue
        self.action_queue = ActionQueue(self.rtc_config)
        
        # Tracking variables
        self.step = 0
        self.total_reward = 0.0
        self.frames = []
        
        # Setup visualization windows
        self._setup_visualization()
    
    def _load_policy(self):
        """Load the SmolVLA policy with RTC configuration."""
        print(f"Loading policy: {self.model_name}")
        
        # Load base configuration
        policy_cfg = PreTrainedConfig.from_pretrained(self.model_name)
        
        # Add RTC configuration
        policy_cfg.rtc_config = self.rtc_config
        
        # Load and configure the policy model
        self.policy = SmolVLAPolicy.from_pretrained(self.model_name, config=policy_cfg)
        self.policy.to(self.device)
        self.policy.eval()  # Set to evaluation mode
        
        print("Policy loaded successfully")
    
    def _setup_environment(self):
        """Initialize the SO-ARM gymnasium environment."""
        print(f"Setting up environment: {self.env_name}")
        
        self.env = gym.make(
            self.env_name,
            render_mode='rgb_array',  # For video recording
            obs_type="pixels_agent_pos"  # Include both images and joint positions
        )
        
        # Reset environment to get initial observation
        self.observation, self.info = self.env.reset()
        
        print("Environment setup complete")
    
    def _setup_processors(self):
        """Load dataset statistics and create pre/post processors."""
        print(f"Loading statistics from: {self.stats_path}")
        
        # Load normalization statistics from training dataset
        stats = cast_stats_to_numpy(load_json(self.stats_path))
        
        # Create processors for normalizing inputs and denormalizing outputs
        self.pre_processor, self.post_processor = make_smolvla_pre_post_processors(
            self.policy.config, 
            stats
        )
        
        print("Processors created successfully")
    
    def _setup_visualization(self):
        """Create OpenCV windows for real-time visualization."""
        cv2.namedWindow('SmolVLA SO-ARM Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SmolVLA SO-ARM Control', 1280, 480)
        
        print("Visualization windows created")
    
    def prepare_observation(self, gym_obs: dict, task_instruction: str) -> dict:
        """
        Convert gym_soarm observation to SmolVLA expected format.
        
        Args:
            gym_obs (dict): Raw observation from gym environment containing:
                - 'pixels': Dict with camera views (diagonal, wrist.right)
                - 'agent_pos': Joint positions in radians (7-dim array)
            
            task_instruction (str): Natural language task description
        
        Returns:
            dict: Preprocessed observation dictionary with keys:
                - 'observation.images.camera1': Normalized diagonal camera (1, C, H, W)
                - 'observation.images.camera2': Normalized wrist camera (1, C, H, W)
                - 'observation.state': Joint positions in degrees (1, N)
                - 'task': List containing task instruction string
        """
        state_obs = gym_obs['agent_pos']
        
        formatted_obs = {
            # Convert images: (H, W, C) -> (1, C, H, W), normalize to [0, 1]
            "observation.images.camera1": torch.from_numpy(
                gym_obs["pixels"]["diagonal"]
            ).permute(2, 0, 1).float().divide(255).unsqueeze(0).to(self.device),
            
            "observation.images.camera2": torch.from_numpy(
                gym_obs['pixels']["wrist.right"]
            ).permute(2, 0, 1).float().divide(255).unsqueeze(0).to(self.device),
            
            # Convert joint positions from radians to degrees, shape: (1, N)
            "observation.state": torch.from_numpy(
                np.rad2deg(state_obs)
            ).float().unsqueeze(0).to(self.device),
            
            # Task must be a list for tokenizer batching
            "task": [task_instruction]
        }
        
        # Apply preprocessing (normalization based on dataset stats)
        return self.pre_processor(formatted_obs)
    
    def create_multi_cam_view(self, gym_obs: dict, step_info: dict) -> np.ndarray:
        """
        Create a composite visualization of multiple camera views with overlays.
        
        Args:
            gym_obs (dict): Observation containing camera images
            step_info (dict): Dictionary with keys:
                - 'step': Current step number
                - 'reward': Current reward value
                - 'status': Status string (e.g., "Executing", "Replanning")
        
        Returns:
            np.ndarray: Composite image (H, 2*W, 3) with both cameras side-by-side
        """
        cam_diagonal = gym_obs["pixels"]["diagonal"]
        cam_wrist = gym_obs["pixels"]["wrist.right"]
        
        h, w = 240, 320
        
        # Resize both camera views
        view1 = cv2.resize(cam_diagonal, (w, h))
        view2 = cv2.resize(cam_wrist, (w, h))
        
        # Stack horizontally
        composite = np.hstack([view1, view2])
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            composite, 
            f"Step: {step_info['step']} | Reward: {step_info['reward']:.2f}", 
            (10, 30), font, 0.6, (0, 255, 0), 2
        )
        cv2.putText(
            composite, 
            f"{step_info['status']}", 
            (10, 60), font, 0.6, (0, 255, 255), 2
        )
        cv2.putText(
            composite, 
            "Diagonal", 
            (10, h-10), font, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            composite, 
            "Wrist Right", 
            (w+10, h-10), font, 0.5, (255, 255, 255), 1
        )
        
        return composite
    
    def create_info_window(self, observation: dict, actions: np.ndarray) -> np.ndarray:
        """
        Create an information panel showing current observations and actions.
        
        Args:
            observation (dict): Current observation with 'agent_pos' key
            actions (np.ndarray): Action array in radians (7-dim)
        
        Returns:
            np.ndarray: Info panel image (600, 400, 3)
        """
        # Create blank panel
        panel = np.ones((600, 400, 3), dtype=np.uint8) * 240
        
        # Title
        cv2.putText(
            panel, "SO-ARM Action Information", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        
        # Observation values (joint positions)
        y_pos = 70
        cv2.putText(
            panel, "Observations", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        
        for i, joint_name in enumerate(JOINTS):
            text = f"{joint_name}: {observation['agent_pos'][i]:.3f} rad"
            cv2.putText(
                panel, text, (20, y_pos + (i + 1) * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )
        
        # Action values
        y_pos1 = y_pos + 220
        cv2.putText(
            panel, "Actions", (20, y_pos1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        
        for i, joint_name in enumerate(JOINTS):
            text = f"{joint_name}: {actions[i]:.3f} rad"
            cv2.putText(
                panel, text, (20, y_pos1 + (i + 1) * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )
        
        return panel
    
    def initialize_action_queue(self):
        """
        Prime the action queue by generating and executing initial actions.
        
        This method:
        1. Generates initial 50-action chunk from the policy
        2. Execute actions till inference_delay
        2. Slice unexecuted actions.
        3. Add the sliced actions to queue.
        
        This priming is necessary for RTC to have execution history when
        it starts blending future plans with past executions.
        """
        print("Initializing action queue by executing first actions...")
        
        # Prepare observation for policy input
        obs_tensor = self.prepare_observation(self.observation, self.task_instruction)
        
        # Generate initial action chunk (50 actions)
        initial_chunk = self.policy.predict_action_chunk(
            batch=obs_tensor,
            noise=torch.randn(1, 50, 32).to(self.device),
            inference_delay=0,  # No previous actions to blend with
            prev_chunk_left_over=None
        )
        
        # Post-process actions (denormalize to degrees)
        with torch.no_grad():
            initial_actions = self.post_processor(initial_chunk)
        
        # Convert to numpy: (1, 50, 6) -> (50, 6)
        current_chunk_actions = initial_actions.squeeze(0).cpu().numpy()
        print(f"Initial chunk generated with {current_chunk_actions.shape[0]} actions")

        # Execute first `inference_delay` actions to prime the system
        for i in range(self.inference_delay):
            action = current_chunk_actions[i]
            
            # Visualize initialization
            step_info = {
                'step': self.step,
                'reward': 0.0,
                'status': 'Initializing'
            }
            composite_view = self.create_multi_cam_view(self.observation, step_info)
            info_view = self.create_info_window(self.observation, np.deg2rad(action))
            
            cv2.imshow('SmolVLA SO-ARM Control', cv2.cvtColor(composite_view, cv2.COLOR_RGB2BGR))
            cv2.imshow("Joint Information", info_view)
            
            if self.record_video:
                self.frames.append(composite_view)
            
            # Execute action in environment (convert degrees to radians)
            self.observation, reward, terminated, truncated, info = self.env.step(
                np.deg2rad(action)
            )
            self.total_reward += reward
            self.step += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
            
            if terminated or truncated:
                break
            
            time.sleep(0.1)  # Brief delay for visualization
        
        print(f"Executed {self.step} initial actions for queue priming")

        # Add unexecuted actions to queue
        unprocessed_actions = initial_chunk[:, self.inference_delay:, :]
        processed_actions = initial_actions[:, self.inference_delay:, :]
        
        self.action_queue._append_actions_queue(
            unprocessed_actions.squeeze(0),
            processed_actions.squeeze(0)
        )
        
        print(f"Queue initialized with {self.action_queue.qsize()} actions")
    
    def run(self):
        """
        Execute the main control loop.
        
        This method:
        1. Initializes the action queue with priming actions
        2. Runs the main loop until max_steps or episode termination
        3. Handles replanning at inference_delay intervals with RTC
        4. Executes actions and renders visualization
        5. Records video frames if enabled
        6. Saves video on completion
        
        Returns:
            dict: Episode statistics containing:
                - 'total_steps': Number of steps executed
                - 'total_reward': Cumulative reward
                - 'terminated': Whether episode ended naturally
                - 'truncated': Whether episode was cut short
        """
        print("Starting simulation...")
        
        # Initialize the action queue
        self.initialize_action_queue()
        
        print("\nStarting main control loop...")
        
        terminated = False
        truncated = False
        
        try:
            while self.step < self.max_steps:
                start_time = time.time()
                
                # Replan every inference_delay steps using RTC
                if self.step % self.inference_delay == 0:
                    status_text = "RTC: Refreshing Plan"
                    
                    # Get current observation
                    obs_tensor = self.prepare_observation(
                        self.observation, 
                        self.task_instruction
                    )
                    
                    # Get leftover actions from previous plan for blending
                    prev_actions = self.action_queue.get_left_over()
                    
                    # Safety check for empty prev_actions
                    if prev_actions is None or \
                       (isinstance(prev_actions, torch.Tensor) and prev_actions.shape[0] == 0):
                        prev_actions = None
                    
                    # Generate new action chunk with RTC blending
                    action_chunk = self.policy.predict_action_chunk(
                        batch=obs_tensor,
                        noise=torch.randn(1, 50, 32).to(self.device),
                        inference_delay=self.inference_delay,
                        prev_chunk_left_over=prev_actions
                    )
                    
                    # Post-process and merge into action queue
                    with torch.no_grad():
                        real_actions = self.post_processor(action_chunk)
                        self.action_queue.merge(
                            action_chunk.squeeze(0),
                            real_actions.squeeze(0),
                            self.inference_delay
                        )
                
                elif (self.step + 1) % self.inference_delay == 0:
                    status_text = "RTC: Refreshing Plan"
                else:
                    status_text = "RTC: Executing Blended Path"
                
                # Get next action from queue
                if self.action_queue.qsize() > 0:
                    current_action_chunk = self.action_queue.get()
                    
                    # Convert tensor to numpy array
                    if isinstance(current_action_chunk, torch.Tensor):
                        current_action_chunk = current_action_chunk.squeeze().detach().cpu().numpy()
                    
                    # Ensure action is 1D
                    if current_action_chunk.ndim != 1:
                        print(f"Warning: Action has shape {current_action_chunk.shape}, taking first element")
                        current_action_chunk = current_action_chunk[0]
                else:
                    print(f"Warning: Queue empty at step {self.step}")
                    # Use zero action as safe fallback
                    current_action_chunk = np.zeros_like(self.observation['agent_pos'])
                
                # Execute action in environment (convert degrees to radians)
                self.observation, reward, terminated, truncated, info = self.env.step(
                    np.deg2rad(current_action_chunk)
                )
                
                # Update visualization
                step_info = {
                    'step': self.step,
                    'reward': reward,
                    'status': status_text
                }
                composite_view = self.create_multi_cam_view(self.observation, step_info)
                info_view = self.create_info_window(
                    self.observation,
                    np.deg2rad(current_action_chunk)
                )
                
                # Display frames
                cv2.imshow('SmolVLA SO-ARM Control', cv2.cvtColor(composite_view, cv2.COLOR_RGB2BGR))
                cv2.imshow("Joint Information", info_view)
                
                # Record frame for video
                if self.record_video:
                    self.frames.append(composite_view)
                
                # Update counters
                self.step += 1
                self.total_reward += reward
                
                # Check termination conditions
                if terminated or truncated:
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Maintain 1 second per step
                elapsed = time.time() - start_time
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)
        
        except KeyboardInterrupt:
            print("\nStopped by user.")
        
        finally:
            self._cleanup()
        
        # Return episode statistics
        return {
            'total_steps': self.step,
            'total_reward': self.total_reward,
            'terminated': terminated,
            'truncated': truncated
        }
    
    def _cleanup(self):
        """Clean up resources and save video."""
        self.env.close()
        cv2.destroyAllWindows()
        
        if self.record_video and self.frames:
            print("Saving video...")
            imageio.mimsave(
                "smolvla_soarm_output.mp4",
                np.stack(self.frames),
                fps=5
            )
            print(f"Video saved. Total steps: {self.step}, Total reward: {self.total_reward:.2f}")


if __name__ == "__main__":
    # Create controller with custom parameters
    controller = SmolVLAController(
        model_name="lerobot/smolvla_base",
        stats_path="svla_so101_pickplace.json",
        env_name='gym_soarm/PickAndPlaceCube-v0',
        task_instruction="blue lego brick",
        execution_horizon=10,  # Blend 10 future steps with past
        max_guidance_weight=5.0,  # RTC guidance strength
        prefix_attention_schedule="LINEAR",  # Exponential attention schedule
        inference_delay=20,  # Replan every 20 steps
        max_steps=200,  # Maximum episode length
        record_video=True,  # Record execution video
        device=None  # Auto-detect GPU/CPU
    )
    
    # Run the control loop
    results = controller.run()
    
    print("\n" + "="*50)
    print("EPISODE COMPLETE")
    print("="*50)
    print(f"Total Steps: {results['total_steps']}")
    print(f"Total Reward: {results['total_reward']:.2f}")
    print(f"Terminated: {results['terminated']}")
    print(f"Truncated: {results['truncated']}")