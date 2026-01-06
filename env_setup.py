#!/usr/bin/env python3
import subprocess
import getpass
import shutil
import sys
import os

from pathlib import Path

def prerequisites():

    commands = [
                ["sudo", "apt", "update"],
                ["sudo", "apt", "upgrade", "-y"],
                ["sudo", "apt", "install", "-y", 
                    "gh", "git", "curl", 
                    "openjdk-17-jdk", "openjdk-17-jre", 
                    "nix", "screen", "cmake"]
            ]
    
    for cmd in commands:
        print(f"Running {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("Command failed:", cmd)
            return -1;
        else:
            print("\n\nPackages installed\n\n")
            return 0;

def lftoolchain():

    curl = subprocess.Popen(
        ["curl", "-Ls", "https://install.lf-lang.org"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        subprocess.run(
            ["bash", "-s", "nightly", "cli"],
            stdin=curl.stdout,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("Installation failed")
        return -1;

    print("\n\nLF toolchain installation successful\n\n")
    return 0

def nixsetup():
    user = getpass.getuser()
    
    nix_dir = Path.home()/".config"/"nix"
    nix_file = nix_dir/"nix.conf"

    commands = [
                ["groups"],
                ["sudo", "usermod", "-aG", "nix-users", user]
            ]
    

    cmd = commands[0]
    result = subprocess.run(cmd, capture_output=True, text=True, check = True)

    if "nix-users" in result.stdout:

        print("\n\nNix users are added\n\n")
        #return 0
    else:
        try: 
            subprocess.run(commands[1], check=True, 
                capture_output=True, 
                text=True
            )

        except subprocess.CalledProcessError as e:
            print("\n\n Failed to set up nix\n\n")
            print("Code:", e.returncode)
            return -1
        
        print("\n\nAdded nix to groups. Reboot needed!!!\n\n")
        #return 0
    nix_dir.mkdir(parents=True, exist_ok=True)

    line = "experimental-features = nix-command flakes\n"

    if nix_file.exists():
        read_lines_nix = nix_file.read_text()
        if line.strip() in read_lines_nix:
            print("\n\nNix experimental-features are already enabled")
    else:
        try:
            with nix_file.open("a") as f:
                f.write(line)
            print("\n\n Nix configured, reboot if needed.\n\n")
        except PermissionError:
            print("Permission error with writing")
        except FileNotFoundError:
            print("Unable to locate file")
        except IsADirectoryError:
            print("That is a directory and not a file!!")
        except OSError as e:
            print(str(e))
        return -1

    return 0

def setup_folder():

    ACT_HOME = Path.home()/"pololu"
    TEST_FOLDER = ACT_HOME/"lf_embedded_lab_testbed/test_lf_programs"
    LF_TEMPLATE = ACT_HOME/"lf-3pi-template"

    os.chdir(ACT_HOME)
    command = [
            ["git", "clone", "git@github.com:lf-lang/lf-3pi-template.git"]]
    cmd = command[0]
    result = subprocess.run(cmd, capture_output=True, text=True, check = True)  

    for file in TEST_FOLDER.glob("*.py"):
        shutil.copy2(file, LF_TEMPLATE)

def main():

    if prerequisites() != 0:
        sys.exit(1)
    
    if lftoolchain() != 0:
        sys.exit(1)
    
    if nixsetup() != 0:
        sys.exit(1)

    setup_folder()

    print("\n\n All prerequisites are installed and packages are ready.\n\n")

if __name__ == "__main__":
    main()
