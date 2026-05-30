$source = "D:\Antigravity Lair\Event Horizont\PUBLIC_RELEASE_R59_CLEAN\ComfyUI-Event-Equality-Core"
$destination = "C:\Users\HYPERPC\Documents\ComfyUI\custom_nodes\ComfyUI-Event-Equality-Core"

Write-Host "Syncing from D: to C:..."

# Copy files, excluding .git, __pycache__, and scratch
robocopy $source $destination /MIR /XD .git __pycache__ scratch /XF sync_to_comfy.ps1

Write-Host "Sync complete! D: drive is the true source of truth."
