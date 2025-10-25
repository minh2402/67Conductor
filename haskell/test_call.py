import subprocess

# Run a bash command and capture its output
def run_bash_command(command):
    try:
        result = subprocess.run(
            command, 
            shell=True,  # Use shell to interpret bash commands
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Return strings instead of bytes
            check=True  # Raise an error if the command fails
        )
        return result.stdout.strip()  # Return cleaned stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

# Example usage
output = run_bash_command("ls -la")
print(output)