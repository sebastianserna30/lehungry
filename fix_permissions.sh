#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    # Use 'set -a' to automatically export variables
    set -a
    source .env
    set +a
else
    echo "Error: .env file not found."
    exit 1
fi

# Function to apply chmod 666
apply_perm() {
    local port=$1
    local name=$2
    if [ ! -z "$port" ]; then
        if [ -e "$port" ]; then
            echo "Applying chmod 666 to $name ($port)..."
            sudo chmod 666 "$port"
        else
            echo "Warning: $name path '$port' does not exist."
        fi
    else
        echo "Info: $name not set in .env"
    fi
}

apply_perm "$LEADER_PORT" "Leader Port"
apply_perm "$FOLLOWER_PORT" "Follower Port"

echo "Permissions updated."
