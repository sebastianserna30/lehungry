import os
from datasets import load_dataset, Dataset, concatenate_datasets
from dotenv import load_dotenv
from openai import OpenAI

# Secure configuration loading
# We explicitly load from .secrets to avoid confusion with other env files
SECRETS_FILE = ".secrets"

if not os.path.exists(SECRETS_FILE):
    print(f"\n[!] CRITICAL ERROR: Secrets file '{SECRETS_FILE}' not found.")
    print("    Please create this file and add your OPENAI_API_KEY and HF_TOKEN.")
    print("    See .secrets.template for an example.")
    exit(1)

load_dotenv(dotenv_path=SECRETS_FILE, override=True)

# Configuration
REPO_ID = "lehungry-robotum/sebas_test" # Default/Fallback
TASK_COLUMN = "single_task"                      # The column to augment
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY or "ENTER_KEY" in OPENAI_API_KEY:
    print(f"\n[!] CRITICAL ERROR: OPENAI_API_KEY is missing or invalid in '{SECRETS_FILE}'.")
    exit(1)

if not HF_TOKEN or "ENTER_KEY" in HF_TOKEN:
    print(f"\n[!] WARNING: HF_TOKEN is possibly missing or invalid in '{SECRETS_FILE}'.")
    print("    Pushing to the hub might fail.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_augmented_tasks(original_task: str, num_augs: int = 3) -> list[str]:
    """
    Uses OpenAI API to generate synonymous task strings.
    """
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not found. Returning empty list.")
        return []

    prompt = (
        f"Generate {num_augs} distinct, natural language variations for the following robot task: '{original_task}'.\n"
        "The variations should convey the exact same meaning but use different words or phrasing suitable for a robot instruction.\n"
        "Return ONLY the variations as a bulleted list, nothing else."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that paraphrases robot commands."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        # robust parsing of the bulleted list
        variations = [line.strip().lstrip('- ').strip() for line in content.split('\n') if line.strip()]
        return variations[:num_augs]

    except Exception as e:
        print(f"Error generating augmentation for '{original_task}': {e}")
        return []

# Global caches
TASK_MAPPING = {}
AUGMENTATION_CACHE = {}

def load_task_mapping(repo_id):
    """
    Downloads and loads the meta/tasks.parquet file to build a task index mapping.
    """
    print(f"Downloading task metadata from {repo_id}...")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        # Check if tasks.parquet exists
        files = list_repo_files(repo_id, repo_type="dataset")
        if "meta/tasks.parquet" not in files:
            print("meta/tasks.parquet not found.")
            return {}

        file_path = hf_hub_download(repo_id=repo_id, filename="meta/tasks.parquet", repo_type="dataset")
        
        # Load using datasets library
        ds = load_dataset("parquet", data_files=file_path, split="train")
        print(f"Tasks metadata columns: {ds.column_names}")
        
        # Assume 'task_index' and 'task' are the columns
        task_col = "task"
        if "task" not in ds.column_names:
            str_cols = [c for c in ds.column_names if isinstance(ds[0][c], str)]
            if str_cols:
                task_col = str_cols[0]
        
        mapping = {row['task_index']: row[task_col] for row in ds}
        print(f"Loaded {len(mapping)} tasks.")
        return mapping
    except Exception as e:
        print(f"Error loading task mapping: {e}")
        return {}

def precompute_augmentations(mapping):
    """
    Generates augmentations for all unique tasks in the mapping.
    """
    print(f"Pre-computing augmentations for {len(mapping)} unique tasks...")
    cache = {}
    for task_id, task_str in mapping.items():
        print(f"Generating synonyms for task {task_id}: '{task_str}'")
        variations = generate_augmented_tasks(task_str)
        # Store original + variations
        cache[task_id] = [task_str] + variations
        print(f" -> Generated {len(variations)} variations.")
    return cache

def augment_batch(batch):
    """
    Iterates over the existing batch and creates new rows for each augmented string.
    """
    # Lists to store all new (original + augmented) data for this batch
    new_columns = list(batch.keys())
    if TASK_COLUMN not in new_columns:
        new_columns.append(TASK_COLUMN)
        
    new_data = {col: [] for col in new_columns}
    
    # Iterate over each example (row) in the batch
    for i in range(len(batch['task_index'])):
        task_idx = batch['task_index'][i]
        
        # Retrieve pre-computed augmented list (original + synonyms)
        # Default to just the original ID string if mapping fails (fallback)
        augmented_tasks = AUGMENTATION_CACHE.get(task_idx, [f"Task {task_idx}"])
        
        # Duplicate the physical data for each augmented task string
        for task_string in augmented_tasks:
            # Copy all features
            for col in batch.keys():
                new_data[col].append(batch[col][i])
            
            # Add the text column
            new_data[TASK_COLUMN].append(task_string)
    
    return new_data

def get_target_dataset():
    """
    Interactively select a dataset from local cache or input a custom name.
    """
    # Check for existing datasets in standard HF cache location
    base_cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot/lehungry-robotum/")
    existing_datasets = []
    
    if os.path.exists(base_cache_dir):
        # List subdirectories
        for name in os.listdir(base_cache_dir):
            if os.path.isdir(os.path.join(base_cache_dir, name)):
                existing_datasets.append(name)
    existing_datasets.sort()
    
    print("\n--- Select Dataset to Augment ---")
    for i, name in enumerate(existing_datasets):
        print(f"{i+1}. lehungry-robotum/{name}")
    print(f"{len(existing_datasets)+1}. Enter custom Repo ID")
    
    repo_id = ""
    while True:
        try:
            choice = input("\nChoice: ").strip()
            if not choice: continue
            
            idx = int(choice) - 1
            if 0 <= idx < len(existing_datasets):
                repo_id = f"lehungry-robotum/{existing_datasets[idx]}"
                break
            elif idx == len(existing_datasets):
                repo_id = input("Enter full Repo ID (e.g. user/dataset): ").strip()
                if repo_id: break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")
            
    return repo_id

def review_augmentations(mapping):
    """
    Interactively generates and reviews augmentations for each task.
    """
    print(f"\n--- Review Augmentations ({len(mapping)} unique tasks) ---")
    cache = {}
    
    for task_id, task_str in mapping.items():
        print(f"\nTask [{task_id}]: '{task_str}'")
        
        while True:
            print("Generating variations...")
            variations = generate_augmented_tasks(task_str)
            
            print("\nProposed variations:")
            for v in variations:
                print(f" - {v}")
            
            choice = input("\nAccept these variations? [y]/n/r (retry): ").strip().lower()
            
            if choice == 'r':
                continue # Retry loop
            elif choice == 'n':
                print("Skipping variations for this task (using original only).")
                cache[task_id] = [task_str]
                break
            else:
                # Default to yes
                cache[task_id] = [task_str] + variations
                break
                
    return cache

def main():
    global TASK_MAPPING, AUGMENTATION_CACHE, REPO_ID
    
    # 1. Select Dataset
    REPO_ID = get_target_dataset()
    print(f"\nSelected Repository: {REPO_ID}")

    print(f"Loading dataset...")
    try:
        # Load the dataset
        dataset = load_dataset(REPO_ID, split='train') 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load task mapping
    TASK_MAPPING = load_task_mapping(REPO_ID)
    if not TASK_MAPPING:
        print("CRITICAL: Could not load task mapping. Aborting.")
        return

    # 3. Interactive Review & Pre-computation
    AUGMENTATION_CACHE = review_augmentations(TASK_MAPPING)

    # 4. Confirm execution
    print(f"\nReady to augment {len(dataset)} episodes.")
    if input("Proceed with dataset generation? [y]/n: ").strip().lower() == 'n':
        print("Aborted.")
        return

    print("Starting augmentation...")
    
    # Apply the augmentation across the entire dataset
    augmented_dataset = dataset.map(
        augment_batch, 
        batched=True, 
        batch_size=1000, 
        remove_columns=dataset.column_names 
    )

    print(f"Original size: {len(dataset)} episodes")
    print(f"Augmented size: {len(augmented_dataset)} episodes")

    # 5. Push to Hub
    target_repo_id = f"{REPO_ID}-augmented"
    print(f"\nTarget Repo ID: {target_repo_id}")
    
    if input("Push to Hugging Face Hub? [y]/n: ").strip().lower() == 'n':
        print("Skipping upload. You can save locally or run again to push.")
        return

    print(f"Pushing to Hub...")
    try:
        augmented_dataset.push_to_hub(
            repo_id=target_repo_id, 
            split='train',
            token=HF_TOKEN or "<YOUR_HF_WRITE_TOKEN>"
        )
        print("Successfully pushed to Hub.")
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        print("Make sure you are authenticated (huggingface-cli login) or have set HF_TOKEN in .env")

if __name__ == "__main__":
    main()
