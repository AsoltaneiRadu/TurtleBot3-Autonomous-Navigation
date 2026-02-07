# 🏎️ TurtleBot3 AutoRace: Autonomous Driving with ROS 2

### Politehnica University of Timișoara (UPT)

This repository contains the implementation of an autonomous driving system for the **TurtleBot3 AutoRace** challenge using **ROS 2**. The robot utilizes computer vision to detect lanes, recognize traffic signs, and navigate a complex track within the **Gazebo** simulation environment.

---

## 🧠 Project Overview

The goal of this project is to develop a self-driving algorithm capable of completing the AutoRace track autonomously. The system processes visual data from the robot's camera to calculate steering angles and velocity in real-time.

### Key Features
* **Lane Following:** Uses computer vision (OpenCV) to detect lane markers (yellow/white lines) and keep the robot centered.
* **Traffic Sign Recognition:** Identifies stop signs, parking signals, and turn indicators.
* **PD/PID Control:** Implements a control loop to smooth out steering and prevent oscillation on curves.
* **Simulation:** Fully integrated with the official TurtleBot3 AutoRace 2020 track in Gazebo.

---

## 🛠️ Technical Architecture

* **Middleware:** ROS 2 (Robot Operating System)
* **Simulation Engine:** Gazebo
* **Language:** Python 3
* **Sensors:** Raspberry Pi Camera Module (Simulated)
* **Algorithm:**
    1.  **Image Preprocessing:** Perspective transform (Bird's Eye View) and HSV color filtering.
    2.  **Error Calculation:** Determines the deviation from the lane center.
    3.  **Control Output:** Generates `Twist` messages (linear/angular velocity) to correct the robot's path.

---

## 🚀 Installation & Setup

### Prerequisites
Ensure you have the following installed on your Ubuntu system:
* ROS 2 (Humble or Foxy)
* TurtleBot3 Packages
* TurtleBot3 Simulations (`turtlebot3_gazebo`)

### Installation
Clone this repository into your ROS 2 workspace:
<img width="373" height="329" alt="Screenshot from 2026-02-03 16-13-48" src="https://github.com/user-attachments/assets/05d5da4f-4d60-4c48-9521-e93d3c2368ce" />

```bash[Screencast from 02-03-2026 04:14:27 PM.webm](https://github.com/user-attachments/assets/2caf180c-b1b6-4358-b408-58dc57441852)[Screencast from 02-03-2026 04:14:27 PM.webm](https://github.com/user-attachments/assets/bfd38a55-b9a1-4ef8-b840-e43686083f21)


cd ~/ros2_ws/src
git clone [https://github.com/AsoltaneiRadu/TurtleBot3-Autonomous-Navigation.git](https://github.com/AsoltaneiRadu/TurtleBot3-Autonomous-Navigation.git)
cd ..
colcon build
source install/setup.bash
🏁 How to Run

Follow these steps to launch the simulation and start the autonomous driver.
Step 1: Launch the Simulation Environment

This command loads the TurtleBot3 Burger model and opens the AutoRace 2020 track in Gazebo, including traffic lights and signs.
Bash

export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_autorace_2020.launch.py

Step 2: Start the Autonomous Driver

Once the simulation is running, open a new terminal and run the driver script. This node processes the camera feed and sends velocity commands to the robot.
Bash

python3 ~/ros2_ws/src/turtlebot3_autorace_driving/turtlebot3_autorace_driving/prof_driver.py
