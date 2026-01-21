#!/usr/bin/env python3
"""
ACT for motors -> test inside a Nix develop shell.
Hardcoded for SensorTiltSolution.lf.
"""

import subprocess
import shlex
import sys
import os

import pandas as pd
from pathlib import Path

import cv2
import numpy as np
import math
import time
import csv

CAMERA_INDEX = 0
RPM = 80
GROUP = 1

HIGH_RESOLUTION = (1920, 1080)
# HSV color ranges (tune these for your setup)
BLUE_LOWER = np.array([90, 80, 80])
BLUE_UPPER = np.array([130, 255, 255])
RED_LOWER1 = np.array([0, 100, 100])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 100, 100])
RED_UPPER2 = np.array([179, 255, 255])

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

# Check what the camera actually accepted- for debug purposes and logs
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

CLEAN_FILE = "blink_results.csv"

def get_centroid(mask):
    moments = cv2.moments(mask)

    #Non-zero area is needed
    if moments["m00"] == 0:
        return None

    #Average position of pixels
    x_center = moments["m10"] / moments["m00"]
    y_center = moments["m01"] / moments["m00"]

    return int(x_center), int(y_center)


def motor_detect():
    prev_angle = None
    rotations = 0
    start_time = time.time()
    elapsed = 0
    rpm = 0
    
    while elapsed < 30:
        ret, frame = cap.read()
        if not ret:
            break
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # Detect blue center
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        blue_center = get_centroid(blue_mask)
    
        # Detect red rotating marker (combine two red hue ranges)
        red_mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        red_mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_center = get_centroid(red_mask)
    
        if blue_center and red_center:
            bx, by = blue_center
            rx, ry = red_center
    
            dx = rx - bx
            dy = ry - by
            angle = math.degrees(math.atan2(dy, dx))
    
            cv2.circle(frame, blue_center, 6, (255, 0, 0), -1)
            cv2.circle(frame, red_center, 6, (0, 0, 255), -1)
            cv2.line(frame, blue_center, red_center, (0, 255, 0), 2)
    
            if prev_angle is not None:
                d_angle = angle - prev_angle
                # Handle wrap-around (e.g., +179 to -179)
                if d_angle < -180:
                    d_angle += 360
                    rotations += 1
    
                ''' TODO: If we are accounting for calculation of speed in counterclockwise.
                elif d_angle > 180:
                    d_angle -= 360
                    rotations -= 1
                '''
                elapsed = time.time() - start_time
                if (rotations > 0) :
                    measured = time.time() - start_time
                    rpm = (abs(rotations) / measured) * 60
                else:
                    rpm = 0
                
                #elapsed = time.time() - start_time
                #rpm = (abs(rotations) / elapsed) * 60
                cv2.putText(frame, f"RPM: {rpm:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
            data = {
            "Group": [GROUP],
            "Actual RPM": [RPM],
            "Measured RPM": [rpm],
            "Time": [elapsed]
            }
        
            df = pd.DataFrame(data)
            csv_filename = "motor.csv"
        
            if os.path.exists(csv_filename):
                df.to_csv(csv_filename, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_filename, mode='w', header=True, index=False)
    
            prev_angle = angle
    
        #cv2.imshow("RPM Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"RPM: {rpm:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()


def clean():
    if os.path.exists(CLEAN_FILE):
        try:
            os.remove(CLEAN_FILE)
            print("Removed file")
        except Exception as e:
            print(f"Failed to remove:{e}")
    else:
        print("File does not exist.")


def build(num):
    #clean()
    
    if num == 1:
        lf_file = "src/Blink_1.lf"
        elf_path = "bin/Blink_1.elf"
    elif num == 2:
        lf_file = "src/Blink_2.lf"
        elf_path = "bin/Blink_2.elf"
    else:
        lf_file = "src/Blink.lf"
        elf_path = "bin/Blink.elf"

    shell_cmd = f"""
        set -e
        echo "Building {lf_file}"
        lfc {shlex.quote(lf_file)}
        echo "Flashing {elf_path}"
        picotool load -x {shlex.quote(elf_path)} -f
        echo "Done"
    """

    try:
        subprocess.run(
            ["nix", "develop", "--command", "bash", "-c", shell_cmd],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\nBuild or flash failed (exit code {e.returncode})")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: 'nix' not found. Make sure Nix is installed and in PATH.")
        sys.exit(1)

def main():

    ACT_HOME = Path.home()/"pololu"
    LF_TEMPLATE = ACT_HOME/"lf-3pi-template"

    os.chdir(LF_TEMPLATE)

    index = 2
    clean()
    while(index):
        build(index)
        motor_detect(index)
        index -= 1
    #plot()
    
if __name__ == "__main__":
    main()
