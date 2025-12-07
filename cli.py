#!/home/sebas/miniforge3/envs/lerobot/bin/python
import os
import lerobot
from huggingface_hub import HfApi
import json
import sys
import serial.tools.list_ports
import time
import subprocess

CONFIG_FILE = "config.json"
ENV_FILE = ".env"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    
    with open(ENV_FILE, 'w') as f:
        for key, value in config.items():
            if key == "leader_port":
                f.write(f'LEADER_PORT="{value}"\n')
                f.write(f'TELEOP_PORT="{value}"\n')
            elif key == "follower_port":
                f.write(f'FOLLOWER_PORT="{value}"\n')
                f.write(f'ROBOT_PORT="{value}"\n')
            else:
                f.write(f'{key.upper()}="{value}"\n')

def get_current_ports_set():
    """Returns a set of device paths (e.g., {'/dev/ttyACM0', ...})"""
    return set(p.device for p in serial.tools.list_ports.comports())

def run_clean_find_port():
    print("\n--- Find Port (Clean Mode) ---")
    print("1. Ensure your device is currently PLUGGED IN.")
    input("   Press Enter to continue...")
    
    initial_ports = get_current_ports_set()
    
    print("\n2. UNPLUG the USB cable now.")
    input("   Press Enter when unplugged...")
    
    after_unplug_ports = get_current_ports_set()
    
    # Calculate difference: ports that were there but are gone now
    removed_ports = initial_ports - after_unplug_ports
    
    found_port = None
    if len(removed_ports) == 1:
        found_port = list(removed_ports)[0]
        print(f"\n---> Detected Port: {found_port}")
    elif len(removed_ports) == 0:
        print("\n[!] No port disappearance detected. Did you unplug it?")
    else:
        print(f"\n[!] Multiple ports disappeared: {removed_ports}")
    
    print("\n3. Please RECONNECT the USB cable now.")
    input("   Press Enter when reconnected...")
    
    # Optional: verify it came back (not strictly necessary for config but good UX)
    # prompt user we are done
    return found_port

def assign_role(port):
    print(f"\nAssign '{port}' to:")
    print("1. Leader (Teleop Arm)")
    print("2. Follower (Robot Arm)")
    print("3. Cancel")
    
    while True:
        choice = input("Choice: ").strip()
        if choice == '1':
            return "leader_port"
        elif choice == '2':
            return "follower_port"
        elif choice == '3':
            return None
        else:
            print("Invalid.")

def run_calibration(config):
    print("\n--- Calibrate Robot ---")
    print("1. Calibrate Leader (Teleop)")
    print("2. Calibrate Follower (Robot)")
    print("3. Cancel")
    
    choice = input("Select robot to calibrate: ").strip()
    
    cmd = []
    
    if choice == '1':
        port = config.get("leader_port")
        if not port:
            print("Error: Leader Port not set. Please 'Find Port' first.")
            return
        print(f"Calibrating LEADER on {port}...")
        cmd = [
            "lerobot-calibrate",
            "--teleop.type=so101_leader",
            f"--teleop.port={port}",
            "--teleop.id=Leader"
        ]
        
    elif choice == '2':
        port = config.get("follower_port")
        if not port:
            print("Error: Follower Port not set. Please 'Find Port' first.")
            return
        print(f"Calibrating FOLLOWER on {port}...")
        cmd = [
            "lerobot-calibrate",
            "--robot.type=so101_follower",
            f"--robot.port={port}",
            "--robot.id=Follower"
        ]
        
    elif choice == '3':
        return
    else:
        print("Invalid choice.")
        return

    try:
        # Run command and allow it to take over stdout/stdin for interaction
        subprocess.run(cmd, check=True)
        print("\nCalibration process finished.")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Calibration failed with exit code {e.returncode}")
    except FileNotFoundError:
        print("\nError: 'lerobot-calibrate' not found in PATH.")

def run_teleop(config):
    print("\n--- Teleoperate Robot ---")
    
    l_port = config.get("leader_port")
    f_port = config.get("follower_port")
    cameras = config.get("robot_cameras", "None")
    
    if not l_port or not f_port:
        print("Error: Both Leader and Follower ports must be set.")
        return

    print(f"Leader:   {l_port}")
    print(f"Follower: {f_port}")
    print(f"Cameras:  {cameras}")
    
    input("\nPress Enter to START teleoperation (Ctrl+C to stop)...")
    
    cmd = [
        "lerobot-teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={f_port}",
        "--robot.id=Follower",
        f"--robot.cameras={cameras}",
        "--teleop.type=so101_leader",
        f"--teleop.port={l_port}",
        "--teleop.id=Leader",
        "--display_data=True"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTeleoperation ended/failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\nTeleoperation stopped.")
    except FileNotFoundError:
        print("\nError: 'lerobot-teleoperate' command not found in PATH.")

def run_record(config):
    print("\n--- Record Dataset ---")
    
    l_port = config.get("leader_port")
    f_port = config.get("follower_port")
    cameras = config.get("robot_cameras", "None")
    
    if not l_port or not f_port:
        print("Error: Both Leader and Follower ports must be set.")
        return

    # Check for existing datasets
    base_cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot/lehungry-robotum/")
    existing_datasets = []
    
    if os.path.exists(base_cache_dir):
        # List subdirectories
        for name in os.listdir(base_cache_dir):
            if os.path.isdir(os.path.join(base_cache_dir, name)):
                existing_datasets.append(name)
    existing_datasets.sort()
    
    # Menu for Resume vs New
    print("Select dataset:")
    for i, name in enumerate(existing_datasets):
        print(f"{i+1}. Resume '{name}'")
    print(f"{len(existing_datasets)+1}. Create NEW dataset")
    print(f"{len(existing_datasets)+2}. Cancel")

    dataset_name = ""
    resume_flag = "false"
    
    while True:
        try:
            choice_idx = int(input("\nChoice: ").strip()) - 1
            if 0 <= choice_idx < len(existing_datasets):
                # Resume existing
                dataset_name = existing_datasets[choice_idx]
                resume_flag = "true"
                print(f"Resuming dataset: {dataset_name}")
                break
            elif choice_idx == len(existing_datasets):
                # New dataset
                dataset_name = input("Enter NEW dataset name (e.g. my_task_test): ").strip()
                if not dataset_name:
                    print("Name cannot be empty.")
                    continue
                resume_flag = "false"
                break
            elif choice_idx == len(existing_datasets) + 1:
                return # Cancel
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")

    # Task and Episodes
    task_desc = input("Enter task description (e.g. 'grab tape'): ").strip()
    if not task_desc:
        task_desc = "generic task" # Fallback
        
    episodes = input("Number of episodes [5]: ").strip()
    if not episodes:
        episodes = "5"
        
    repo_id = f"lehungry-robotum/{dataset_name}"
    
    print("\n-------------------------------------------")
    print(f"Repo ID: {repo_id}")
    print(f"Task:    {task_desc}")
    print(f"Resume:  {resume_flag}")
    print("-------------------------------------------")
    
    input("Press Enter to START recording (Ctrl+C to stop)...")
    
    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={f_port}",
        "--robot.id=Follower",
        f"--robot.cameras={cameras}",
        "--teleop.type=so101_leader",
        f"--teleop.port={l_port}",
        "--teleop.id=Leader",
        "--display_data=true",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.single_task={task_desc}",
        f"--resume={resume_flag}"
    ]
    
    try:
        subprocess.run(cmd, check=True)

        if resume_flag == "false":
            try:
                # Use the actual lerobot version
                version = f"v{lerobot.__version__}"
                repo_type = "dataset"
                print(f"\n[TAGGING] Creating tag '{version}' for repo '{repo_id}'...")
                
                hub_api = HfApi()
                hub_api.create_tag(repo_id=repo_id, tag=version, repo_type=repo_type)
                print(f"[TAGGING] Successfully tagged {repo_id} with {version}")
            except Exception as e:
                print(f"\n[TAGGING] Warning: Failed to create tag. Error: {e}")
    except subprocess.CalledProcessError as e:
        print(f"\nRecording ended/failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    except FileNotFoundError:
        print("\nError: 'lerobot-record' command not found in PATH.")

def main_menu():
    config = load_config()
    
    while True:
        print("\n==============================")
        print("   LE ROBOT CLI COMMANDER")
        print("==============================")
        l_port = config.get("leader_port", "Not Set")
        f_port = config.get("follower_port", "Not Set")
        
        print(f"Leader Port:   {l_port}")
        print(f"Follower Port: {f_port}")
        print("------------------------------")
        print("1. Find Port")
        print("2. Calibrate Robot")
        print("3. Teleoperate")
        print("4. Record Dataset")
        print("q. Quit")
        
        choice = input("\nSelect an option: ").strip().lower()

        if choice == '1':
            port = run_clean_find_port()
            
            if port:
                role_key = assign_role(port)
                if role_key:
                    config[role_key] = port
                    save_config(config)
                    print(f"[{role_key.replace('_', ' ').title()}] set to {port}")
                    print(f"Saved to {CONFIG_FILE} and {ENV_FILE}")

        elif choice == '2':
            run_calibration(config)
            
        elif choice == '3':
            run_teleop(config)

        elif choice == '4':
            run_record(config)

        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
