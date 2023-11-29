import pandas as pd
import matplotlib.pyplot as plt
import os
import platform
import subprocess
import shutil
import json
from datetime import datetime

s_label = 'Stomach'
j_label = 'Jejunum'
l_label = 'Ileum'
g_label = 'Blood Glucose'
i_label = 'Blood Insulin'

class OpenAPS:

    def __init__(self, recorded_carbs = None, autosense_ratio = 1.0, test_timestamp = "2023-01-01T18:00:00-00:00", profile_path = None) -> None:
        self.shell = "Windows" in platform.system()
        oref_help = subprocess.check_output(["oref0", "--help"], shell=self.shell)

        if "oref0 help - this message" not in str(oref_help):
            print("ERROR - oref0 not installed")
            exit(1)

        if profile_path == None:
            self.profile_path = os.environ["profile_path"]
        else:
            self.profile_path = profile_path
        self.basal_profile_path = os.environ["basal_profile_path"]
        self.autosense_ratio = autosense_ratio
        self.test_timestamp = test_timestamp
        self.epoch_time = int(datetime.strptime(test_timestamp, "%Y-%m-%dT%H:%M:%S%z").timestamp() * 1000)
        self.pump_history = []
        self.recorded_carbs = recorded_carbs

    def run(self, model_history, output_file = None, faulty = False):
        if output_file == None:
            output_file = './openaps_temp'
        
        if not os.path.exists(output_file):
            os.mkdir(output_file)

        time_since_start = len(model_history) - 1
        current_epoch = self.epoch_time + 60000 * time_since_start
        current_timestamp = datetime.fromtimestamp(current_epoch / 1000).strftime("%Y-%m-%dT%H:%M:%S%z")

        basal_history = []
        temp_basal = '{}'
        if model_history[0][i_label] > 0:
            basal_history.append(f'{{"timestamp":"{datetime.fromtimestamp(self.epoch_time/1000).strftime("%Y-%m-%dT%H:%M:%S%z")}"' +
                                 f',"_type":"Bolus","amount":{model_history[0][i_label] / 1000},"duration":0}}')

        for idx, (rate, duration, timestamp) in enumerate(self.pump_history):
            basal_history.append(f'{{"timestamp":"{timestamp}","_type":"TempBasal","temp":"absolute","rate":{str(rate)}}}')
            basal_history.append(f'{{"timestamp":"{timestamp}","_type":"TempBasalDuration","duration (min)":{str(duration)}}}')
            if idx == len(self.pump_history) - 1:
                if faulty:
                    temp_basal = f'{{"duration": {duration}, "temp": "absolute", "rate": {str(rate)}}}'
                else:
                    temp_basal_epoch = int(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp() * 1000)
                    if (current_epoch - temp_basal_epoch) / 60 <= duration:
                        temp_basal = f'{{"duration": {duration}, "temp": "absolute", "rate": {str(rate)}}}'
        basal_history.reverse()

        glucose_history = []
        carb_history = []
        for idx, time_step in enumerate(model_history):
            if idx % 5 == 0:
                bg_level = int(time_step[g_label])
                new_time_epoch = self.epoch_time + idx * 60000
                new_time_stamp = datetime.fromtimestamp(new_time_epoch/1000).strftime("%Y-%m-%dT%H:%M:%S%z")
                glucose_history.append(f'{{"date":{new_time_epoch},"dateString":"{new_time_stamp}","sgv":{bg_level},' +
                                       f'"device":"fakecgm","type":"sgv","glucose":{bg_level}}}')
                
            if idx == 0:
                if time_step[s_label] > 0:
                    if self.recorded_carbs == None:
                        carb_history.append(f'{{"enteredBy":"fakecarbs","carbs":{time_step[s_label]},"created_at":"{self.test_timestamp}","insulin": null}}')
                    else:
                        carb_history.append(f'{{"enteredBy":"fakecarbs","carbs":{self.recorded_carbs},"created_at":"{self.test_timestamp}","insulin": null}}')

            else:
                carb_diff = time_step[s_label] - model_history[idx - 1][s_label]
                if carb_diff > 0:
                    new_time_epoch = self.epoch_time + idx * 60000
                    new_time_stamp = datetime.fromtimestamp(new_time_epoch/1000).strftime("%Y-%m-%dT%H:%M:%S%z")
                    carb_history.append(f'{{"enteredBy":"fakecarbs","carbs":{time_step[s_label]},"created_at":"{new_time_stamp}","insulin":null}}')
        glucose_history.reverse()
        carb_history.reverse()

        self.__make_file_and_write_to(f"{output_file}/clock.json", f'"{current_timestamp}-00:00"')
        self.__make_file_and_write_to(f"{output_file}/autosens.json", '{"ratio":' + str(self.autosense_ratio) + '}')
        self.__make_file_and_write_to(f"{output_file}/pumphistory.json", "[" + ','.join(basal_history) + "]")
        self.__make_file_and_write_to(f"{output_file}/glucose.json", "[" + ','.join(glucose_history) + "]")
        self.__make_file_and_write_to(f"{output_file}/carbhistory.json", "[" + ','.join(carb_history) + "]")
        self.__make_file_and_write_to(f"{output_file}/temp_basal.json", temp_basal)

        iob_output = subprocess.check_output([
            "oref0-calculate-iob",
            f"{output_file}/pumphistory.json",
            self.profile_path,
            f"{output_file}/clock.json",
            f"{output_file}/autosens.json"
        ], shell=self.shell, stderr=subprocess.DEVNULL).decode("utf-8")
        self.__make_file_and_write_to(f"{output_file}/iob.json", iob_output)

        meal_output = subprocess.check_output([
            "oref0-meal",
            f"{output_file}/pumphistory.json",
            self.profile_path,
            f"{output_file}/clock.json",
            f"{output_file}/glucose.json",
            self.basal_profile_path,
            f"{output_file}/carbhistory.json"
        ], shell=self.shell, stderr=subprocess.DEVNULL).decode("utf-8")
        self.__make_file_and_write_to(f"{output_file}/meal.json", meal_output)

        basal_res = subprocess.run([
            "oref0-determine-basal",
            f"{output_file}/iob.json",
            f"{output_file}/temp_basal.json",
            f"{output_file}/glucose.json",
            self.profile_path,
            "--auto-sens",
            f"{output_file}/autosens.json",
            "--meal",
            f"{output_file}/meal.json",
            "--microbolus",
            "--currentTime",
            str(current_epoch)
        ], shell=self.shell, capture_output=True, text=True)
        
        if "Warning: currenttemp running but lastTemp from pumphistory ended" in basal_res.stdout:
            shutil.rmtree(output_file, ignore_errors=True)
            return -1

        self.__make_file_and_write_to(f"{output_file}/suggested.json", basal_res.stdout)        

        json_output = open(f"{output_file}/suggested.json")
        data = json.load(json_output)

        rate = data["rate"] if "rate" in data else 0
        if rate != 0:
            duration = data["duration"]
            timestamp = data["deliverAt"]
            self.pump_history.append((rate, duration, timestamp))

        shutil.rmtree(output_file, ignore_errors=True)

        return 1000 * rate / 60.0

    def __make_file_and_write_to(self, file_path, contents):
        file = open(file_path, "w")
        file.write(contents)

class Model:
    def __init__(self, starting_vals, constants):
        self.interventions = dict()

        self.history = []
        self.history.append({'step': 0, 
                             s_label: starting_vals[0], 
                             j_label: starting_vals[1], 
                             l_label: starting_vals[2], 
                             g_label: starting_vals[3], 
                             i_label: starting_vals[4]})

        self.kjs = constants[0]
        self.kgj = constants[1]
        self.kjl = constants[2]
        self.kgl = constants[3]
        self.kxg = constants[4]
        self.kxgi = constants[5]
        self.kxi = constants[6]

        self.tau = constants[7]
        self.klambda = constants[8]
        self.eta = constants[9]

        self.gprod0 = constants[10]
        self.kmu = constants[11]
        self.gb = starting_vals[3]

        self.gprod_limit = (self.klambda * self.gb + self.gprod0 * (self.kmu + self.gb)) / (self.klambda + self.gprod0)


    def update(self, t):
        old_s = self.history[t-1][s_label]
        old_j = self.history[t-1][j_label]
        old_l = self.history[t-1][l_label]
        old_g = self.history[t-1][g_label]
        old_i = self.history[t-1][i_label]

        new_s = old_s - (old_s * self.kjs)

        new_j = old_j + (old_s * self.kjs) - (old_j * self.kgj) - (old_j * self.kjl)

        phi = 0 if t < self.tau else self.history[t - self.tau][j_label]
        new_l = old_l + (phi * self.kjl) - (old_l * self.kgl)

        g_prod = (self.klambda * (self.gb - old_g))/(self.kmu + (self.gb - old_g)) + self.gprod0 if old_g <= self.gprod_limit else 0
        new_g = old_g - (self.kxg + self.kxgi * old_i) * old_g + g_prod + self.eta * (self.kgj * old_j + self.kgl * old_l)
        
        new_i = old_i - (old_i  * self.kxi)

        if t in self.interventions:
            for intervention in self.interventions[t]:
                if intervention[0] == s_label:
                    new_s += intervention[1]
                elif intervention[0] == i_label:
                    new_i += intervention[1]

        timestep = {'step': t, s_label: new_s, j_label: new_j, l_label: new_l, g_label: new_g, i_label: new_i}
        self.history.append(timestep)

        return [old_s, new_s, old_j, new_j, old_l, new_l, old_g, new_g, old_i, new_i, g_prod, 
                self.kjs, self.kgj, self.kjl, self.kgl, self.kxg, self.kxgi, self.kxi, self.tau,
                self.klambda, self.eta, self.gprod0, self.kmu, self.gb]

    def add_intervention(self, timestep: int, variable: str, intervention: float):
        if timestep not in self.interventions:
            self.interventions[timestep] = list()

        self.interventions[timestep].append((variable, intervention))

    def plot(self, timesteps = -1):
        if timesteps == -1:
            df = pd.DataFrame(self.history)
            df.plot('step', [s_label, j_label, l_label, g_label, i_label])
            plt.show()
        else:
            df = pd.DataFrame(self.history[:timesteps])
            df.plot('step', [s_label, j_label, l_label, g_label, i_label])
            plt.show()