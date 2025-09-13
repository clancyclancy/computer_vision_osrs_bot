from collections import Counter, deque,defaultdict
import time
import math
import copy
#import pyautogui

class Action:
    def __init__(self):
        self.desired_object = None
        self.action_type    = None #"Left_Click", "Right_Click", "Space_Bar"

        self.action_click_location = (0,0)
        self.action_vertical_offset = 0 #click on center or some offset from center

       # self.secondary_action = False
       # self.secondary_action_type = None #"Left_Click", "Right_Click", "Space_Bar"
        #self.secondary_desired_object = None

        # actions like banking, stairs, require precise clicks, thererfore player must not move.
        #moving between waypoints is imprecise and chill, so can be moving and clicking.
        #self.must_be_stationary_for_action = True
        #self.must_be_stationary_for_next_action = True





class Bot:
    def __init__(self):
        # Constructor: initializes the object
        self.is_initialized = False

        self.desired_object = None
        self.temporary_desired_object = None

        #self.occasional_click_timer = 15 #seconds
        #self.last_occasional_click = 0   #seconds

        self.current_state = None


        self.objects = None

        self.Action = None

        self.raw_fish_count    = deque(maxlen=100)
        self.cooked_fish_count = deque(maxlen=50)
        self.instantaneous_raw_fish_count = 0
        self.instantaneous_cooked_fish_count = 0


        self.distance_history = defaultdict(lambda: deque(maxlen=30))

        self.cooking = False
        self.fishing = False
        self.traveling = False
        self.last_travel_time = float("inf")
        self.last_command_time = float("inf")

        self.bank_clicked = False
        self.bankDepositAttempts = 0
        #self.traveling_to_bank_have_not_taken_stairs = False
        #self.traveling_to_fish_have_not_taken_stairs = False
        self.upstairs = False
        #self.number_bank_deposit_actions = 0

        #self.must_be_stationary_for_action = True

        #don't spam click the same object again and again
        #self.last_object_clicked = None

        self.LastAction = Action()
        self.LastAction_cpa = float("inf")
        self.cpa_threshold = 125

        self.screen_width = 1920
        self.screen_height = 1080

        self.increase_tile_costs = False
        self.CLICK_TYPE = {
        "minimap":       "Left_Click",
        "bank":          "Left_Click",
        "player":        "",
        "fishing_spot":  "Left_Click",
        "raw":           "Right_Click",
        "cooked":        "Right_Click",
        "cooking_range": "Left_Click",
        "stairs":        "Right_Click",
        "interact_up":   "Left_Click",
        "interact_down": "Left_Click",
        "textbox":       "Space_Bar",
        "bank_screen":   "",
        "bank_deposit":  "Left_Click"
        }
        self.CLICK_Y_SHIFT = {
        "bank":          0,
        "player":        0,
        "fishing_spot":  0,
        "raw":           0,
        "cooked":        0,
        "cooking_range": 0,
        "stairs":        0,
        "interact_up":   0.43,
        "interact_down": 0.42,
        "textbox":       0,
        "bank_screen":   0,
        "bank_deposit":  0.65
        }

        self.NEXT_DESIRED_OBJECT = {
        "bank":          "stairs",
        "player":        None,
        "fishing_spot":  "cooking_range",
        "raw":           None,
        "cooked":        "bank_deposit",
        "cooking_range": "stairs",
        "stairs":        "interact_", # yeah i don't loveeeeeee this
        "interact_up":   "bank",
        "interact_down": "fishing_spot",
        "textbox":       None,
        "bank_screen":   None,
        "bank_deposit":  "stairs"
        }

        self.FULL_INVENTORY = 26


        self.current_state = None

        #self.position_relative_to_desired_object = '' #yeah this sucks, but I made two unique objects the same color so this is the quick fix to avoid 3 more hours of object labeling

    def initialize(self):
        ''' determine the current state of the bot based on detected objects
            initialization states:
            deposit cooked fish in bank
            cook fish at range
            collect fish at river
        '''
        # Filter by confidence threshold
        threshold = 0.4
        filtered = [obj for obj in self.objects if obj['confidence'] >= threshold]

        # Count labels
        label_counts = Counter([obj['label'] for obj in filtered])
        print(label_counts)

        raw_fish_count = label_counts.get('raw_fish', 0)
        cooked_fish_count = label_counts.get('cooked_fish', 0)

        if cooked_fish_count == 0 and raw_fish_count < 26: #assume some bad detections, max is 26
            #go fishing
            self.desired_object = 'fishing_spot'

            #self.desired_goal  = 'fish'

            self.increase_tile_costs = True

        elif raw_fish_count > 0:
            #go cooking
            self.desired_object = 'cooking_range'
            #self.desired_goal  = 'cook'

            self.increase_tile_costs = False
        else:
            #go deposit
            self.desired_object = 'bank_deposit'
            #self.desired_goal  = 'bank'

            self.increase_tile_costs = False

    def record_info(self):
        # track info
        self.get_fish_trend()
        self.is_player_traveling()

        if self.LastAction_cpa > self.cpa_threshold:
            if self.LastAction.desired_object is not None:
                cpa = self.get_distance_to_object(self.LastAction.desired_object["label"])
                print("calc'd cpa")
                print(cpa)
            else:
                cpa = float("Inf")
            self.LastAction_cpa = min(self.LastAction_cpa,cpa)



    def determine_action2(self):

        commanded_action = self.determine_action_helper()
        #print("deterining_action")
        #print(commanded_action)
        #print(commanded_action.action_type)
        #print(commanded_action.desired_object)
        if commanded_action.action_type is None or commanded_action.desired_object is None:
            return Action()

        #is the action something that requires the player to be stationary before clicking?
        # print('stationary check')
        # print(commanded_action.action_type)
        # print(commanded_action)
        # print(commanded_action.desired_object["label"])
        if commanded_action.desired_object["label"] in {"stairs_yellow"}:
            #is the player stationary?
            if self.traveling:
                #wait until not traveling
                print("Canceling Action: movement -------------------------------")
                return Action()


        #is the player at a fishing spot, clicked on a fishing spot, but isnt "fishing" probably because they have caught a fish?
        if self.LastAction.action_type is not None:
            if self.LastAction.desired_object["label"] == "fishing_cyan":
                if self.is_at_desired_location():
                    if (time.time() - self.last_command_time) < 5:
                        print("Canceling Action: fish spam -------------------------------")
                        return Action()

        if self.LastAction.action_type is not None:
            if commanded_action.desired_object["label"] in {"waypoint_blue", "waypoint_green", "waypoint_pink", "interact_orange","stairs_yellow","fishing_cyan"}:
                if self.LastAction.desired_object["label"] == commanded_action.desired_object["label"]:
                    if self.traveling or self.fishing or self.cooking or (time.time() - self.last_travel_time) < 1 or (time.time() - self.last_command_time) < 1:
                        # don't spam the same command if the command is related
                        # print("Canceling Action: spam -------------------------------")
                        # print(self.traveling)
                        # print(self.fishing)
                        # print(self.cooking)
                        # print((time.time() - self.last_travel_time))
                        # print((time.time() - self.last_command_time))
                        return Action()

        #get the distance from the player to the object from the last action
        #player has to be close-ish to last desired object to issue new command if last desired object was a waypoint
        if self.traveling or self.last_travel_time + 2 > time.time(): # only care about CPA when in motion
            if self.LastAction.action_type is not None:
                if commanded_action.desired_object["label"] in {"waypoint_blue", "waypoint_green", "waypoint_pink"}:
                    #print("Previous distance and new distance")
                    #print(self.LastAction_cpa)
                    #cpa = self.get_distance_to_object(self.LastAction.desired_object["label"])
                    #self.LastAction_cpa =  min(self.LastAction_cpa,cpa)
                    # print("Last Action object and Distance------------------")
                    # print(self.LastAction.desired_object["label"])
                    # print(self.LastAction_cpa)
                    # print("Current Action")
                    # print(commanded_action.desired_object["label"])
                    if self.LastAction_cpa >  self.cpa_threshold:
                        print("Canceling Action: CPA -------------------------------")
                        return Action()

        if commanded_action.action_type != "Space_Bar":
            self.LastAction_cpa = float("inf")
            self.last_command_time = time.time()
            self.LastAction = commanded_action
        return commanded_action

    def determine_action_helper(self):

        commanded_action = self.determine_action_helper()
        if commanded_action.action_type is None:
            return Action()


        if commanded_action.desired_object == "fishing_spot":
            self.current_state = "fishing"
        elif commanded_action.desired_object == "cooking_range":
            self.current_state = "cooking"
        elif commanded_action.desired_object == "bank_deposit":
            self.current_state = "banking"



        #send action
        # only change when action is carried out
        self.desired_object = self.NEXT_DESIRED_OBJECT.get(self.desired_object)
        return commanded_action


    def determine_action_helper(self):

        #define Action class
        self.Action = Action()


        # Transition from current state?
        self.current_state = None
        if self.current_state == "fishing":
            #1. inventory full?
            self.get_fish_trend()
            if self.instantaneous_cooked_fish_count >= self.FULL_INVENTORY:
                # Finished fishing, go cook
                self.current_state = "cooking"
                self.desired_object = "cooking_range"

            #2. next to fishing spot?
            dist = self.get_distance_to_object("fishing_spot")
            if dist < self.cpa_threshold:
                return None



            #3. fishing?




        #0. Text Box Visible
        if any(obj["label"] == "textbox" for obj in self.objects):
            self.Action.action_type = "Space_Bar"
            self.Action.desired_object = None#next((obj for obj in self.objects if obj["label"] == "textbox"), None)
            return self.Action


        #1. Looking for desired object

        if any(obj["label"] in self.desired_object for obj in self.objects):
            self.Action.action_type    = self.CLICK_TYPE.get(self.desired_object)
            self.Action.desired_object = self.desired_object
            desired_obj = next((obj for obj in self.objects if obj["label"] == self.desired_object), None)

            (x1, y1, x2, y2) = desired_obj["bbox"]

            height = y2-y1
            width = x2-x1
            center_x = x1 + width/2
            center_y = y1 + height/2

            percentage_shift_y = self.CLICK_Y_SHIFT.get(self.desired_object)

            self.Action.action_click_location = (center_x, center_y + percentage_shift_y * height)


            #only change when action is carried out
            self.desired_object = self.NEXT_DESIRED_OBJECT.get(self.desired_object)




        return None

    def determine_action_helper2(self):
        '''
        Determine where to go based on available information

        '''


        #define Action class
        self.Action = Action()


        # 0.0 CLICKED ON BANK AND ARE WAITING TO DEPOSIT
        if self.bank_clicked:
            if not any(obj["label"] == "interact_orange" for obj in self.objects):
                # just clicked on the bank booth, and the orange interact disappeared... therefore the bank screen opened
                print("in bank screen----------------------------------------")

                if self.instantaneous_cooked_fish_count > 0:
                    print("depositing fish----------------------------------")
                    # deposit cooked fish
                    self.Action.desired_object = self.get_northern_most_specific_object("cooked_fish")
                    print("northern point")
                    print(self.Action.desired_object)
                    self.Action.action_type = "Right_Click"

                    if self.LastAction.action_type is not None:
                        # multiple deposits
                        self.bankDepositAttempts += 1
                    if self.bankDepositAttempts < 4:
                        self.apply_secondary_action(y_shift=int(0.1575*self.screen_height))
                    else:
                        if self.bankDepositAttempts % 2 == 0:
                            self.apply_secondary_action(y_shift=0)
                        else:
                            self.apply_secondary_action(y_shift=int(0.025*self.screen_height))

                    return self.Action
                else:
                    #finished banking, go fish
                    print("finish banking-------------------------------------------------")
                    self.bank_clicked = False
                    self.bankDepositAttempts = 0
                    self.desired_goal = 'fish'
                    self.desired_object = 'fishing_cyan'
                    self.general_direction = "north"
                    #self.traveling_to_fish_have_not_taken_stairs = True
            else:
                #clicked bank, but can still see orange, probably walking?
                return self.Action



        #0. IF THERE IS A TEXTBOX, PRESS SPACE TO CLEAR IT -------------------------------------------------------------
        if any(obj["label"] == "textbox" for obj in self.objects):
            self.Action.action_type = "Space_Bar"
            self.Action.desired_object = self.objects[1]
            return self.Action


        #print("looking for desired state------------------------------")
        #1. LOOK FOR AND CAN SEE DESIRED STATE -------------------------------------------------------------------------
        # if looking for fishing spot and can see fishing spot
        if self.desired_object == "fishing_cyan" and any(obj["label"] == "fishing_cyan" for obj in self.objects):
            # See fishing spot and want to fish
            if self.is_at_desired_location():
                #check if inventory is full
                if (self.instantaneous_raw_fish_count + self.instantaneous_cooked_fish_count) >= 26:
                    #full inventory
                    self.desired_object = 'interact_orange'
                    self.desired_goal = 'cook'
                    self.general_direction = 'south'
                    return self.Action #return nothing this instant
                #only click once in a while
                #if self.last_occasional_click + self.occasional_click_timer < time.time():
                if not self.fishing:
                    # click
                    #return next(obj for obj in self.objects if obj["label"] == "fishing_cyan")
                    self.Action.action_type = "Left_Click"
                    self.Action.desired_object = self.get_closest_specific_object()
                    return self.Action
                else:
                    return self.Action
            else:
                # not at fishing spot
                # click
                self.Action.action_type = "Left_Click"
                self.Action.desired_object = self.get_closest_specific_object()
                return self.Action

        #print("looking for orange interact")
        # look for and can orange interact tile
        # see object interact object
        if self.desired_object == "interact_orange" and any(obj["label"] == "interact_orange" for obj in self.objects):
            #print("interact_orange detected")
            # orange interact object is cooking range
            if self.is_orange_interact_cooking_range():
                #print("determined cooking range")
                # want to cook
                if self.desired_goal == "cook":
                    #print("want to cook")

                    # next to cooking_range
                    if self.is_at_desired_location():
                        #print("at desired loc")

                        #cooking?
                        if self.cooking:
                            #actively cooking, do nothing
                            return self.Action
                        else:
                            #not actively cooking, is there raw fish?
                            if self.instantaneous_raw_fish_count > 0 and self.instantaneous_cooked_fish_count < 26: #sum(1 for obj in self.objects if obj["label"] == "raw_fish") > 0:
                                self.Action.action_type = "Left_Click"
                                self.Action.desired_object = next(obj for obj in self.objects if obj["label"] == "interact_orange")
                                return self.Action
                            else:
                                # no raw fish
                                # want to bank
                                self.desired_goal = "bank"
                                self.general_direction = "south"

                                #self.traveling_to_bank_have_not_taken_stairs = True
                    else:
                        #print("not at desired location")
                        self.Action.action_type = "Left_Click"
                        self.Action.desired_object = next(obj for obj in self.objects if obj["label"] == "interact_orange")
                        return self.Action


            else:
               # print("determined bank")

                if self.desired_goal == "bank":
                    #print("want to go to bank")
                    #click on bank
                    self.bank_clicked = True
                    self.Action.action_type = "Left_Click"
                    self.Action.desired_object = next(obj for obj in self.objects if obj["label"] == "interact_orange")
                    return self.Action


        #2 STAIRS
        #print("looking for stairs-------------------------------")
        if any(obj["label"] == "stairs_yellow" for obj in self.objects):
            #print("stairs yellow detected")
            #print("upstairs?")
            #print(self.is_upstairs_at_bank())
            #print(self.traveling_to_bank_have_not_taken_stairs)
            if not self.is_upstairs_at_bank() and self.desired_goal == "bank":
                #print("want to go to bank, havent taken stairs")
                #go up the stairs
                #self.traveling_to_bank_have_not_taken_stairs = False
                self.Action.desired_object = next((obj for obj in self.objects if obj["label"] == "stairs_yellow"), None)
                self.Action.action_type = "Right_Click"
                self.apply_secondary_action(y_shift=int(0.025*self.screen_height))
                return self.Action

            elif self.is_upstairs_at_bank() and self.desired_goal == "fish":
                #print("want to go fish, havent taken stairs")
                #go down the stairs
                #self.traveling_to_fish_have_not_taken_stairs = False
                self.Action.desired_object = next((obj for obj in self.objects if obj["label"] == "stairs_yellow"), None)
                self.Action.action_type = "Right_Click"
                self.apply_secondary_action(y_shift=int(0.07*self.screen_height))
                return self.Action


        #3 TRAVEL TO DESIRED STATE
        print("looking for waypoints------------------------------")

        # Step 1: Filter for bounding boxes labeled "waypoint"
        waypoints = [obj for obj in self.objects if "waypoint" in obj["label"]]

        # Step 2: Sort by y_min (northmost = closest to y=0)
        # Assuming bbox format is [x_min, y_min, x_max, y_max]
        waypoints_sorted = sorted(waypoints, key=lambda obj: obj["bbox"][1])  # y_mi
        print(waypoints_sorted)

        if len(waypoints_sorted) > 0:
            self.Action.action_type = "Left_Click"
            if self.general_direction == "north":
                self.Action.desired_object = waypoints_sorted[0] #north most waypoint
            else:
                self.Action.desired_object = waypoints_sorted[-1] #south most waypoint
            return self.Action
        return self.Action



    def is_orange_interact_cooking_range(self):
        '''
        I messed up and have both the cooking range and the bank labeled orange.
        This provides a problem when figuring out which is which
        If both green and orange boxes are in frame, then the orange is the cooking range
        I hate this but don't want to redo hours of object labeling
        '''

        labels_present = {obj["label"] for obj in self.objects}
        required_labels = {"interact_orange", "waypoint_green"}

        return required_labels.issubset(labels_present)

    def get_distance_to_object(self,specified_object):
        print("enter get_distance_to_object------")
        player = next((obj for obj in self.objects if obj["label"] == "player_red"), None)
        desired_objects = [obj for obj in self.objects if obj["label"] == specified_object]
        min_dist = float("inf")
        # print(player)
        # print(specified_object)
        # #print(self.objects)
        # for obj in desired_objects:
        #     print("in desired_objects loop")
        #     print(obj)
        # print("here---------")
        # prevent Error
        if player is None or not desired_objects:
            print("no desired object detected, returning inf---------------------")
            return min_dist

        # find the desired_objects closest to the player
        print("before calc loop")
        for obj in desired_objects:
            print("in calc loop")
            #print(obj)
            dist = get_distance(player["bbox"], obj["bbox"])
            if dist < min_dist:
                print("found new min")
                print(min_dist)
                min_dist = dist
        #print("after calc loop")
        return min_dist



    def get_closest_specific_object(self):

        player = next((obj for obj in self.objects if obj["label"] == "player_red"), None)
        desired_objects = (obj for obj in self.objects if obj["label"] == self.desired_object)

        #prevent Error
        if player is None or desired_objects is None:
            return None

        #find the desired_objects closest to the player
        closest = None
        min_dist = float("inf")


        for obj in desired_objects:
            dist = get_distance(player["bbox"], obj["bbox"])
            if dist < min_dist:
                min_dist = dist
                closest = obj
        return closest

    def get_northern_most_specific_object(self,desired_object):
        print("get north")
        print(desired_object)

        desired_objects = (obj for obj in self.objects if obj["label"] == desired_object)
        print(desired_objects)
        #prevent Error
        if desired_objects is None:
            return None

        #find the desired_objects closest
        closest = None
        min_dist = float("inf")

        for obj in desired_objects:
            x1, _, x2, _ = obj["bbox"]
            print("bbox coords")
            print(x1)
            print(x2)
            dist = get_distance((x1, 0, x2, 0), obj["bbox"])
            print(dist)
            if dist < min_dist:
                print("new min")

                min_dist = dist
                closest = obj
        return closest


    def is_at_desired_location(self):

        # determine location of player
        player = next((obj for obj in self.objects if obj["label"] == "player_red"),None)
        closest = self.get_closest_specific_object()
        if player is None or closest is None:
            return False

        if get_distance(player["bbox"],closest["bbox"]) < 50: #less than 50 is right next to, less than 75 is diagonal
            return True

        return False

    def get_fish_trend(self):
        # Record timestamp + count
        self.instantaneous_raw_fish_count = sum(1 for obj in self.objects if obj["label"] == "raw_fish")
        self.instantaneous_cooked_fish_count = sum(1 for obj in self.objects if obj["label"] == "cooked_fish")
        self.raw_fish_count.append(self.instantaneous_raw_fish_count)
        self.cooked_fish_count.append(self.instantaneous_cooked_fish_count)


        # if len(self.raw_fish_count) >= 5:  #
        #     total_count = 0
        #     for _,count in self.raw_fish_count:
        #             total_count += count
        #     if self.instantaneous_raw_fish_count > (total_count/len(self.raw_fish_count)+1):
        #         self.fishing = True
        #     else:
        #         self.fishing = False

        # if len(self.cooked_fish_count) >= 5:  # wait until we have a few points
        #     total_count = 0
        #     for _,count in self.cooked_fish_count:
        #             total_count += count
        #     if self.instantaneous_cooked_fish_count > (total_count/len(self.cooked_fish_count)+1):
        #         self.cooking = True
        #     else:
        #         self.cooking = False
        if len(self.raw_fish_count) >= 6:
            if sum(list(self.raw_fish_count)[:5]) / 5 + 0.1 < sum(list(self.raw_fish_count)[-5:]) / 5:
                self.fishing = True
            else:
                self.fishing = False

        if len(self.cooked_fish_count) >= 6:
            if sum(list(self.cooked_fish_count)[:5]) / 5 + 0.1 < sum(list(self.cooked_fish_count)[-5:]) / 5:
                self.cooking = True
            else:
                self.cooking = False


    def apply_secondary_action(self, y_shift=100):

        self.Action.secondary_action = True
        self.Action.secondary_action_type = "Left_Click"
        self.Action.secondary_desired_object = copy.deepcopy(self.Action.desired_object) # i hate OOP
        #self.Action.secondary_desired_object = self.Action.desired_object

        # shift mouse down for secondary click
        coords = self.Action.secondary_desired_object["bbox"]
        x1, y1, x2, y2 = coords
        new_coords = (x1, y1 + y_shift, x2, y2 + y_shift)

        self.Action.secondary_desired_object["bbox"] = new_coords
        #print("primary action coords")
        #print(self.Action.desired_object["bbox"])
        #print("secondary action coords")
        #print(self.Action.secondary_desired_object["bbox"])


    def is_player_traveling(self):
        #print("determine if player is traveling")
        # 1) Identify player_red
        player = next((obj for obj in self.objects if obj["label"] == "player_red"), None)
        if player is None:
            self.traveling = False  # no player detected
            return

        player_center = get_center(player["bbox"])

        # 2) Count occurrences of each label
        label_counts = defaultdict(int)
        for obj in self.objects:
            label_counts[obj["label"]] += 1

        # 3) Iterate unique detections
        for obj in self.objects:
            if label_counts[obj["label"]] == 1 and obj["label"] != "player_red" and obj["confidence"] >= 0.6:
                obj_center = get_center(obj["bbox"])
                dist = math.dist(player_center, obj_center)

                # Save distance history for this object label
                self.distance_history[obj["label"]].append(dist)

                # 4) Check movement trend
                if len(self.distance_history[obj["label"]]) >= 5:
                    #print("computing movement trend")
                    # Compare the latest distance to average of previous values
                    prev_avg = sum(list(self.distance_history[obj["label"]])[:-1]) / (
                                len(self.distance_history[obj["label"]]) - 1)
                    #print(prev_avg)
                    latest = self.distance_history[obj["label"]][-1]
                    #print(latest)
                    delta = latest - prev_avg
                    #print("average delta of bbboxes")
                   # print(delta)
                    #print(self.distance_history)

                    if abs(delta) > 10:  # pixels threshold for movement
                        self.traveling = True
                        self.last_travel_time = time.time()
                        #print(f"[Movement] Player appears to be moving relative to {obj['label']}: Δ = {delta:.1f}px")

                    else:
                        self.traveling = False

    def is_upstairs_at_bank(self):
        '''

        '''
        if len(self.objects) > 0:
            labels_present = {obj["label"] for obj in self.objects}
            required_labels = {"stairs_yellow", "waypoint_green"}
            return not required_labels.issubset(labels_present)

        return True #idk w/e





def get_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2, (y_min + y_max) / 2

def get_distance(bbox1, bbox2):
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5





