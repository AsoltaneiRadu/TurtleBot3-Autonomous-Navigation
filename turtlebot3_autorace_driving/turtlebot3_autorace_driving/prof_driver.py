import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time

# --- STATE DEFINITIONS ---
STATE_WAITING = 0
STATE_DRIVING = 1
STATE_INTERSECTION = 2
STATE_PARKING = 3
STATE_BARRIER = 4

class ProfDriver(Node):
    def __init__(self):
        super().__init__('prof_driver_node')
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
            
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.br = CvBridge()
        
        # --- VARIABLES ---
        self.current_state = STATE_WAITING
        self.start_time = time.time()
        self.last_error = 0
        self.lane_width = 420 
        self.scan_ranges = []
        
        # --- TRAFFIC LIGHT VARIABLES ---
        self.traffic_light_detected = False
        
        # --- INTERSECTION VARIABLES ---
        self.intersection_step = 0
        self.intersection_timer = 0
        self.intersection_done = False
        self.intersection_finish_time = 0
        
        # --- PARKING VARIABLES ---
        self.parking_step = 0
        self.parking_timer = 0
        self.parking_done = False
        self.barrier_finish_time = 0
        self.barrier_step = 0
        self.barrier_timer = 0
        
        # --- OBSTACLE AVOIDANCE MEMORY ---
        self.avoidance_state_active = False
        self.avoidance_cooldown_time = 0
        
        # --- LOAD TEMPLATES ---
        self.template_right = self.load_sign_template('right_turn.png', 'blue')
        self.template_parking = self.load_sign_template('parking3.png', 'blue')
        self.template_stop = self.load_sign_template('stop.png', 'red')

        # --- MEMORY ---
        self.last_sign_type = None
        self.last_sign_time = 0

        self.get_logger().info('Burger: Systems Online! 🍔')
        self.get_logger().info('Burger: Gata de cursa! 🍔🏎️')

    def load_sign_template(self, filename, color_filter='blue'):
        # 1. Aflam unde este scriptul pe hard disk
        script_dir = os.path.dirname(os.path.realpath(__file__))
        img_path = os.path.join(script_dir, filename)

        self.get_logger().info(f"📂 Caut imaginea la adresa: {img_path}")

        # 2. Incercam sa incarcam (COLOR FIRST)
        loaded_img = cv2.imread(img_path)

        # 3. VERIFICAREA
        if loaded_img is None:
            self.get_logger().error(f"❌ EROARE CRITICA: Imaginea '{filename}' NU a fost gasita!")
            return None
        
        # --- SMART AUTO-CROP (BLUE FILTER) ---
        # Filter the template for BLUE to isolate the sign from the background/pole
        hsv_temp = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2HSV)
        
        if color_filter == 'blue':
            lower_blue = np.array([95, 100, 50])
            upper_blue = np.array([145, 255, 255])
            mask_temp = cv2.inRange(hsv_temp, lower_blue, upper_blue)
        else: # Red
            lower_red1 = np.array([0, 100, 50]); upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 50]); upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv_temp, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_temp, lower_red2, upper_red2)
            mask_temp = cv2.bitwise_or(mask1, mask2)
        
        cnts_temp, _ = cv2.findContours(mask_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_template = None
        if cnts_temp:
            c = max(cnts_temp, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped_temp = loaded_img[y:y+h, x:x+w]
            self.get_logger().info(f"✂️ Template {filename} Auto-Cropped to Blue Area: {w}x{h}")
            
            # Convert to Gray and Equalize (to match live processing)
            gray_temp = cv2.cvtColor(cropped_temp, cv2.COLOR_BGR2GRAY)
            final_template = cv2.equalizeHist(gray_temp)
        else:
            self.get_logger().warn(f"⚠️ No Blue found in {filename}! Using full image.")
            gray_temp = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2GRAY)
            final_template = cv2.equalizeHist(gray_temp)
            
        return final_template

    def scan_callback(self, msg):
        if len(self.scan_ranges) == 0:
            self.get_logger().info(f"✅ LiDAR Connected! Ranges: {len(msg.ranges)}")
        self.scan_ranges = msg.ranges

    def birds_eye_view(self, img):
        height, width = img.shape[:2]
        # Burger Camera Calibration (Low Camera)
        p1 = [0, height]
        p2 = [width, height]
        p3 = [width * 0.80, height * 0.50]
        p4 = [width * 0.20, height * 0.50]
        
        src_points = np.float32([p1, p2, p3, p4])
        dst_points = np.float32([
            [width * 0.25, height],
            [width * 0.75, height],
            [width * 0.75, 0],
            [width * 0.25, 0]
        ])

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(img, M, (width, height))

    def detect_traffic_light_start(self, cv_image):
        # Look at the top center of the image
        height, width = cv_image.shape[:2]
        roi = cv_image[0:int(height/3), int(width/3):int(width*2/3)]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for Green
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
        
        if np.sum(mask_green > 0) > 100:
            return True
        return False

    def search_for_sign(self, cv_image):
        if self.template_right is None and self.template_parking is None and self.template_stop is None: return None

        # Use the raw image (cv_image) which is the correct POV for vertical signs.
        # (The lane follower uses the warped bird's eye view).
        
        # 1. Convert to HSV for Color Detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # 2. Define Blue Range (for the Blue Sign Background)
        # Hue: 95-145 (Narrower Blue), Saturation: >100, Value: >50
        lower_blue = np.array([95, 100, 50])
        upper_blue = np.array([145, 255, 255])
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 2b. Define Red Range (for Stop Sign)
        lower_red1 = np.array([0, 100, 50]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 50]); upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Morphological operations to remove noise
        kernel = np.ones((3,3), np.uint8)
        mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
        mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)
        mask_red = cv2.erode(mask_red, kernel, iterations=1)
        mask_red = cv2.dilate(mask_red, kernel, iterations=2)

        # 3. Find Contours on the Mask
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_img = cv_image.copy()
        detected_type = None
        
        # --- PROCESS BLUE SIGNS ---
        for cnt in contours_blue:
            area = cv2.contourArea(cnt)
            
            # Filter small noise
            if area < 500: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # Check if it's roughly square (Sign shape)
            if 0.3 < aspect_ratio < 1.6:
                # --- SHAPE ANALYSIS (Square vs Circle) ---
                # Square fills the box (~0.9-1.0), Circle leaves corners empty (~0.78)
                rect_area = w * h
                extent = area / float(rect_area)
                
                # 4. Template Matching Verification
                # Extract the ROI (Region of Interest)
                roi = cv_image[y:y+h, x:x+w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.equalizeHist(roi_gray) # Improve contrast for better matching
                
                max_val_right = 0
                max_val_park = 0

                # --- CHECK RIGHT TURN ---
                if self.template_right is not None:
                    t_h, t_w = self.template_right.shape
                    roi_resized = cv2.resize(roi_gray, (t_w, t_h))
                    res = cv2.matchTemplate(roi_resized, self.template_right, cv2.TM_CCOEFF_NORMED)
                    _, max_val_right, _, _ = cv2.minMaxLoc(res)

                # --- CHECK PARKING ---
                if self.template_parking is not None:
                    t_h, t_w = self.template_parking.shape
                    roi_resized = cv2.resize(roi_gray, (t_w, t_h))
                    res = cv2.matchTemplate(roi_resized, self.template_parking, cv2.TM_CCOEFF_NORMED)
                    _, max_val_park, _, _ = cv2.minMaxLoc(res)
                
                # --- MEMORY UPDATE (Track signs even if far away) ---
                # If we see a sign clearly (even if small), remember it!
                if max_val_right > 0.5:
                    self.last_sign_type = "right"
                    self.last_sign_time = time.time()
                elif max_val_park > 0.5:
                    self.last_sign_type = "parking"
                    self.last_sign_time = time.time()

                # --- HYBRID SCORING (Template + Shape) ---
                # Base score is the Template Match (P vs Arrow)
                score_right = max_val_right
                score_park = max_val_park
                
                # Shape Bias: Square (>0.80) boosts Parking, Circle (<0.75) boosts Right
                # Increased weights to prevent misclassification
                if extent > 0.80: score_park += 0.5 # Huge bonus for Square
                elif extent < 0.75: score_right += 0.3 # Bonus for Circle
                
                # Draw Debug Info
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_img, f"E:{extent:.2f} SR:{score_right:.2f} SP:{score_park:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # 5. Stop Condition
                # If the best score is good enough (>0.3) AND is close enough (area > 1500)
                if area > 1500:
                    if score_park > score_right and score_park > 0.3:
                        detected_type = "parking"
                        cv2.putText(debug_img, "PARKING!", (x, y+h+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        break
                    
                    elif score_right > score_park and score_right > 0.3:
                        detected_type = "right"
                        cv2.putText(debug_img, "RIGHT TURN!", (x, y+h+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        break
                    
                    # --- MEMORY FALLBACK ---
                    # If we are close (area > 2000) but the match is bad (e.g. cut off),
                    # check if we saw it clearly recently (< 2.0s ago).
                    elif time.time() - self.last_sign_time < 2.0:
                        detected_type = self.last_sign_type
                        cv2.putText(debug_img, f"MEMORY: {detected_type}", (x, y+h+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                        self.get_logger().info(f"🧠 Memory Trigger: {detected_type}")
                        break

        # --- PROCESS RED SIGNS (STOP) ---
        if detected_type is None: 
            for cnt in contours_red:
                area = cv2.contourArea(cnt)
                if area < 500: continue
                
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                
                # Stop sign is octagonal, roughly square bounding box
                if 0.5 < aspect_ratio < 1.5:
                    roi = cv_image[y:y+h, x:x+w]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.equalizeHist(roi_gray)
                    
                    if self.template_stop is not None:
                        t_h, t_w = self.template_stop.shape
                        roi_resized = cv2.resize(roi_gray, (t_w, t_h))
                        res = cv2.matchTemplate(roi_resized, self.template_stop, cv2.TM_CCOEFF_NORMED)
                        _, max_val_stop, _, _ = cv2.minMaxLoc(res)
                        
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(debug_img, f"STOP: {max_val_stop:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        # Dynamic Threshold: If score is very high (>0.65), accept smaller signs (further away)
                        # Otherwise, require standard size (>1500)
                        if (max_val_stop > 0.65 and area > 800) or (max_val_stop > 0.45 and area > 1500):
                            detected_type = "stop"
                            cv2.putText(debug_img, "STOP SIGN!", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            break
                        
        cv2.imshow("Sign Mask", mask_blue)
        cv2.imshow("Red Mask", mask_red)
        cv2.imshow("Sign Detection View", debug_img)
        
        return detected_type

    def follow_lane_logic(self, cv_image):
        height, width, _ = cv_image.shape
        warped_image = self.birds_eye_view(cv_image)
        hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
        
        # Lane Masks
        lower_yellow = np.array([20, 100, 100]); upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        lower_white = np.array([0, 0, 150]); upper_white = np.array([179, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        scan_y = int(height * 0.85)
        
        def get_center(mask):
            px = np.where(mask[scan_y, :] > 0)[0]
            return int(np.mean(px)) if len(px) > 0 else None

        cy = get_center(mask_yellow)
        cw = get_center(mask_white)
  

        img_center = width // 2
        path_center = img_center 
        
        # Dual Lane Logic
        if cy is not None and cw is not None: path_center = (cy + cw) / 2
        elif cy is not None: path_center = cy + (self.lane_width / 2)
        elif cw is not None: path_center = cw - (self.lane_width / 2)

        # --- FIX DIRECTION ---
        # Daca linia e in stanga (path_center < image_center), eroarea e pozitiva.
        # Angular Z pozitiv = Viraj Stanga. ACUM E CORECT!
        error = img_center - path_center 
        
        Kp = 0.005; Kd = 0.008
        derivative = error - self.last_error
        self.last_error = error
        lane_angular_z = (error * Kp) + (derivative * Kd)
        
        final_angular_z = lane_angular_z
        final_linear_x = 0.15 - (abs(lane_angular_z) * 0.12)
        if final_linear_x < 0.05: final_linear_x = 0.05
        
        # --- COOLDOWN LOGIC ---
        if not self.avoidance_state_active and (time.time() - self.avoidance_cooldown_time < 1.0):
            final_angular_z = 0.0
            final_linear_x = 0.12
            self.last_error = 0
        
        # --- OBSTACLE AVOIDANCE (Prioritize Track + Verification) ---
        override_by_obstacle = False
        if len(self.scan_ranges) > 0:
            ranges = np.array(self.scan_ranges)
            ranges[ranges == 0.0] = 999.0
            ranges[np.isinf(ranges)] = 999.0
            
            # 1. Check obstacles in the direction of the Green Point (path_center)
            # error = img_center - path_center (Positive = Left)
            # Map error to degrees. Max angle approx 30 deg.
            look_angle_deg = int(((img_center - path_center) / img_center) * 30)
            sector_size = 15 # +/- 15 degrees around the target path
            
            if len(ranges) >= 360:
                indices = np.arange(look_angle_deg - sector_size, look_angle_deg + sector_size) % 360
                min_dist = np.min(ranges[indices])
                
                # 2. Hysteresis for entering/exiting avoidance mode
                if min_dist < 0.28: # Enter avoidance if something is closer than 28cm
                    self.avoidance_state_active = True
                elif min_dist > 0.50 and self.avoidance_state_active: # Exit avoidance once path is clear for 50cm
                    self.avoidance_state_active = False
                    self.avoidance_cooldown_time = time.time()
                
                # 3. If in avoidance mode, take over control
                if self.avoidance_state_active:
                    override_by_obstacle = True
                    self.get_logger().info(f"--- OBSTACLE AVOIDANCE ACTIVE (Dist: {min_dist:.2f}m) ---")
                    
                    # --- Emergency Stop ---
                    # If we are about to crash, stop moving forward and turn sharply.
                    if min_dist < 0.22:
                        self.get_logger().warn("🚨 EMERGENCY STOP & TURN!")
                        final_linear_x = 0.0
                        # Find largest gap to turn towards
                        left_gap = np.mean(ranges[30:90]) # Check 30-90 degrees left
                        right_gap = np.mean(ranges[-90:-30]) # Check 30-90 degrees right
                        if left_gap > right_gap:
                            final_angular_z = 2.5 # Sharp turn left
                        else:
                            final_angular_z = -2.5 # Sharp turn right
                    
                    # --- Standard Arcing Maneuver ---
                    # Otherwise, perform a steady arc to get around the obstacle.
                    else:
                        # Decide which way to go based on the largest gap
                        left_gap = np.mean(ranges[20:80])
                        right_gap = np.mean(ranges[-80:-20])
                        
                        if left_gap > right_gap:
                            # More space on the left, so arc left
                            self.get_logger().info(f"Arcing LEFT (L:{left_gap:.1f}m > R:{right_gap:.1f}m)")
                            final_angular_z = 1.8 # Constant left turn
                        else:
                            # More space on the right, so arc right
                            self.get_logger().info(f"Arcing RIGHT (R:{right_gap:.1f}m > L:{left_gap:.1f}m)")
                            final_angular_z = -1.8 # Constant right turn
                        
                        # Move forward slowly to execute the arc
                        final_linear_x = 0.05

        twist = Twist()
        twist.angular.z = float(final_angular_z)
        twist.linear.x = float(final_linear_x)
        self.publisher.publish(twist)
        
        # Debug Lane View
        debug_warp = warped_image.copy()
        if override_by_obstacle:
            cv2.putText(debug_warp, "AVOIDING OBSTACLE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif time.time() - self.avoidance_cooldown_time < 1.0:
            cv2.putText(debug_warp, "COOLDOWN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        else:
            cv2.circle(debug_warp, (int(path_center), scan_y), 8, (0, 255, 0), -1) 
        cv2.imshow("Driving View", debug_warp)

    def image_callback(self, msg):
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e: return

        # --- STATE MACHINE ---
        
        if self.current_state == STATE_WAITING:
            self.publisher.publish(Twist()) # Stop
            
            is_green = self.detect_traffic_light_start(cv_image)
            elapsed = time.time() - self.start_time
            
            # Start if Green OR Timeout (5 seconds)
            if is_green or elapsed > 5.0:
                self.current_state = STATE_DRIVING
                self.get_logger().info("GO GO GO! 🟢🏎️")
            else:
                 # Optional: Show waiting view
                 pass

        elif self.current_state == STATE_DRIVING:
            # 1. Drive
            self.follow_lane_logic(cv_image)
            
            # 2. Look for Signs
            sign_type = self.search_for_sign(cv_image)
            
            if sign_type == "right" and not self.intersection_done:
                self.get_logger().info("INTERSECTION DETECTED! STOPPING! 🛑")
                self.current_state = STATE_INTERSECTION
                self.intersection_turn_direction = "right"
                self.intersection_step = 0
                self.intersection_timer = time.time()
                self.publisher.publish(Twist()) # Hard Stop
            elif sign_type == "parking" and not self.parking_done:
                # Check Cooldown: Don't detect parking immediately after an intersection turn
                if self.intersection_done and (time.time() - self.intersection_finish_time < 4.0):
                    self.get_logger().info("⏳ Ignoring Parking Sign (Cooldown)...")
                else:
                    self.get_logger().info("PARKING SIGN DETECTED! STOPPING! 🅿️")
                    self.current_state = STATE_PARKING
                    self.parking_step = 0
                    self.parking_timer = time.time()
                    self.publisher.publish(Twist()) # Hard Stop
            elif sign_type == "stop":
                if time.time() - self.barrier_finish_time < 5.0:
                    self.get_logger().info("⏳ Ignoring Stop Sign (Cooldown)...")
                else:
                    self.get_logger().info("STOP SIGN DETECTED! APPROACHING BARRIER... 🛑")
                    self.current_state = STATE_BARRIER
                    self.barrier_step = 0
                    self.barrier_timer = time.time()
        
        elif self.current_state == STATE_INTERSECTION:
            # Step 0: Stop for 2 seconds
            if self.intersection_step == 0:
                self.publisher.publish(Twist())
                self.search_for_sign(cv_image) # Keep drawing debug box
                
                if time.time() - self.intersection_timer > 2.0:
                    self.get_logger().info("TURNING RIGHT... ↪️")
                    self.intersection_step = 1
                    self.intersection_timer = time.time()
            
            # Step 1: Turn 90 degrees
            elif self.intersection_step == 1:
                twist = Twist()
                twist.angular.z = -0.5 # Negative is Right
                self.publisher.publish(twist)
                
                # Turn for ~3.14 seconds (90 deg at 0.5 rad/s)
                if time.time() - self.intersection_timer > 3.14:
                    self.get_logger().info("TURN COMPLETE! RESUMING TRACK! 🏎️")
                    self.current_state = STATE_DRIVING
                    self.last_error = 0 # Reset PID error
                    self.intersection_done = True
                    self.intersection_finish_time = time.time()
                    
        elif self.current_state == STATE_PARKING:
            # --- ORIGINAL PARKING MANEUVER ---
            
            # Step 0: Drive Forward for 5.0 seconds (Align with spot)
            if self.parking_step == 0:
                twist = Twist()
                twist.linear.x = 0.1
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 5.0:
                    self.get_logger().info("PARKING: ALIGNED! TURNING LEFT... ⬅️")
                    self.parking_step = 1
                    self.parking_timer = time.time()
            
            # Step 1: Turn Left 90 degrees
            elif self.parking_step == 1:
                twist = Twist()
                twist.angular.z = 0.5 # Positive is Left
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 3.5: # ~100 degrees
                    self.get_logger().info("PARKING: TURN DONE! DRIVING FORWARD... ⬆️")
                    self.parking_step = 2
                    self.parking_timer = time.time()
            
            # Step 2: Drive Forward for 10 seconds (Into the spot)
            elif self.parking_step == 2:
                twist = Twist()
                twist.linear.x = 0.1
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 10.0:
                    self.get_logger().info("PARKING: SPOT REACHED! STARTING EXIT MANEUVER... ↪️")
                    self.parking_step = 3
                    self.parking_timer = time.time()

            # --- NEW MANEUVER (EXIT/COMPLEX) ---

            # Step 3: Turn Right 90 degrees
            elif self.parking_step == 3:
                twist = Twist()
                twist.angular.z = -0.5 # Negative is Right
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 3.14: # ~90 degrees
                    self.get_logger().info("PARKING: TURN RIGHT DONE! FORWARD... ⬆️")
                    self.parking_step = 4
                    self.parking_timer = time.time()
            
            # Step 4: Go Forward for 3 seconds
            elif self.parking_step == 4:
                twist = Twist()
                twist.linear.x = 0.1
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 3.0:
                    self.get_logger().info("PARKING: STOPPING FOR 5s... 🛑")
                    self.parking_step = 5
                    self.parking_timer = time.time()
            
            # Step 5: Stop for 5 seconds
            elif self.parking_step == 5:
                self.publisher.publish(Twist())
                
                if time.time() - self.parking_timer > 5.0:
                    self.get_logger().info("PARKING: GOING BACK... ⬇️")
                    self.parking_step = 6
                    self.parking_timer = time.time()

            # Step 6: Go Back for 3 seconds
            elif self.parking_step == 6:
                twist = Twist()
                twist.linear.x = -0.1
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 3.0:
                    self.get_logger().info("PARKING: BACK DONE! TURN RIGHT... ↪️")
                    self.parking_step = 7
                    self.parking_timer = time.time()

            # Step 7: Turn Right 90 degrees again
            elif self.parking_step == 7:
                twist = Twist()
                twist.angular.z = -0.5 # Right
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 3.14:
                    self.get_logger().info("PARKING: TURN 2 DONE! FORWARD... ⬆️")
                    self.parking_step = 8
                    self.parking_timer = time.time()

            # Step 8: Go Forward for 10 seconds
            elif self.parking_step == 8:
                twist = Twist()
                twist.linear.x = 0.1
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 9.0:
                    self.get_logger().info("PARKING: EXITING SPOT... TURN 90... ↪️")
                    self.parking_step = 9
                    self.parking_timer = time.time()

            # Step 9: Turn 90 degrees (Left) to resume track
            elif self.parking_step == 9:
                twist = Twist()
                twist.angular.z = 0.5 # Left
                self.publisher.publish(twist)
                
                if time.time() - self.parking_timer > 3.14:
                    self.get_logger().info("PARKING COMPLETE! RESUMING TRACK! 🏎️")
                    self.current_state = STATE_DRIVING
                    self.parking_done = True
                    self.intersection_done = False

        elif self.current_state == STATE_BARRIER:
            # Step 0: Keep moving for 2 seconds to pass the sign/approach barrier
            if self.barrier_step == 0:
                self.follow_lane_logic(cv_image)
                
                if time.time() - self.barrier_timer > 2.0:
                    self.get_logger().info("WAITING FOR BARRIER... 🛑")
                    self.barrier_step = 1
                    self.barrier_timer = time.time()
            
            # Step 1: Stop and wait for the barrier to open
            elif self.barrier_step == 1:
                self.publisher.publish(Twist()) # Stop
                
                # --- LIDAR BARRIER DETECTION ---
                barrier_open = False
                
                if len(self.scan_ranges) > 0:
                    ranges = np.array(self.scan_ranges)
                    # Handle 0.0 and inf as "far away" (no obstacle)
                    ranges[ranges == 0.0] = 999.0
                    ranges[np.isinf(ranges)] = 999.0
                    
                    # Check Front Sector (-20 to +20 degrees)
                    arc = 20
                    if len(ranges) >= 360:
                        front_ranges = np.concatenate((ranges[:arc], ranges[-arc:]))
                        min_dist = np.min(front_ranges)
                        
                        self.get_logger().info(f"🔍 LiDAR Front Dist: {min_dist:.2f}m")
                        
                        # If closest obstacle is far enough (> 2.30m), barrier is open
                        if min_dist > 2.30:
                            barrier_open = True
                
                # Timeout Check (10 seconds)
                if time.time() - self.barrier_timer > 10.0:
                    self.get_logger().info("BARRIER TIMEOUT! FORCING START! 🟢⏰")
                    self.current_state = STATE_DRIVING
                    self.barrier_finish_time = time.time()
                    self.last_sign_type = None
                
                # LiDAR Condition
                elif barrier_open:
                    self.get_logger().info("BARRIER OPEN (LiDAR)! RESUMING TRACK! 🟢")
                    self.current_state = STATE_DRIVING
                    self.barrier_finish_time = time.time()
                    self.last_sign_type = None # Reset memory

        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    driver = ProfDriver()
    try: rclpy.spin(driver)
    except KeyboardInterrupt: pass
    finally: driver.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()