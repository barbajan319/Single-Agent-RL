import numpy as np
from numpy import random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class ChargingEnv(gym.Env):
    def __init__(self, price=1):
        super().__init__()
        self.numbers_of_cars = 10
        self.number_of_days = 1
        self.price_flag = price
        self.done = False

        EV_Capacity = 350
        charging_effic = 0.91
        discharging_effic = 0.91
        charging_rate = 180
        discharging_rate = 180
        self.info = {}
        self.max_capacity = charging_rate * (self.numbers_of_cars * 0.8)
        self.EV_param = {
            "charging_effic": charging_effic,
            "EV_capacity": EV_Capacity,
            "discharging_effic": discharging_effic,
            "charging_rate": charging_rate,
            "discharging_rate": discharging_rate,
        }

        low = np.array(np.zeros(4 + 2 * self.numbers_of_cars), dtype=np.float32)
        high = np.array(np.ones(4 + 2 * self.numbers_of_cars), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.numbers_of_cars,), dtype=np.float32
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed

    def step(self, actions):

        [reward, Grid, Cost_EV, Battery_Degradation] = self.simulate_control(actions)
        self.Grid_Evol.append(Grid)
        # self.Penalty_Evol.append(Cost_EV)
        self.Reward.append(reward)
        self.Cost_History.append(Cost_EV)
        self.Battery_Degradation.append(Battery_Degradation)
        self.timestep += 1
        self.truncation = False
        conditions = self.get_obs()
        if self.timestep == 24:
            self.test = True
            self.done = True
            self.timestep = 0
            self.Results = {
                "SOC": self.SOC,
                "Grid_Final": self.Grid_Evol,
                "Cost_History": self.Cost_History,
                "Cost_Reward": self.Reward,
                "Battery_Degradation": self.Battery_Degradation,
            }

        return conditions, -reward, self.done, self.truncation, self.info

    def reset(self, seed=0):
        self.timestep = 0
        self.day = 1
        self.price_flag = random.randint(1, 4)
        self.test = False
        self.done = False
        self.SOC_FINISH = []
        ArrivalT = []
        DepartureT = []
        SOC = np.zeros([self.numbers_of_cars, 25])
        present_cars = np.zeros([self.numbers_of_cars, 25])

        for car in range(self.numbers_of_cars):
            present = 0
            pointer = 0
            Arrival_car = []
            Departure_car = []

            for hour in range(24):
                if present == 0:
                    arrival = round(random.rand() - 0.1)
                    if arrival == 1 and hour <= 20:
                        ran = random.randint(20, 50)
                        SOC[car, hour] = ran / 100
                        pointer = pointer + 1
                        Arrival_car.append(hour)
                        upper_limit = min(hour + 10, 25)
                        Departure_car.append(random.randint(hour + 4, int(upper_limit)))

                if arrival == 1 and pointer > 0:
                    if hour < Departure_car[pointer - 1]:
                        present = 1
                        present_cars[car, hour] = 1
                    else:
                        present = 0
                        present_cars[car, hour] = 0
                else:
                    present = 0
                    present_cars[car, hour] = 0

            ArrivalT.append(Arrival_car)
            DepartureT.append(Departure_car)

        evolution_of_cars = np.zeros([24])
        for hour in range(24):
            evolution_of_cars[hour] = np.sum(present_cars[:, hour])

        self.Price = self.Energy_calculation()
        self.initial = {
            "SOC": SOC,
            "ArrivalT": ArrivalT,
            "evolution_of_cars": evolution_of_cars,
            "DepartureT": DepartureT,
            "present_cars": present_cars,
        }

        return (self.get_obs(), self.info)

    def get_obs(self):
        if self.timestep == 0:
            self.Cost_History = []
            self.Reward = []
            self.Grid_Evol = []
            self.Penalty_Evol = []
            self.Battery_Degradation = []
            self.SOC = self.initial["SOC"]

        [self.leave, Departure_hour, Battery] = self.simulate_station()
        current_price = np.array(self.Price[0, self.timestep] / 0.1)
        future_prices = np.array(self.Price[0, self.timestep + 1 : self.timestep + 4])
        states = np.concatenate(
            (np.array(Battery, dtype=np.float32), np.array(Departure_hour) / 24),
            axis=None,
        )
        observations = np.concatenate(
            (current_price, future_prices, states), axis=None
        ).astype(np.float32)

        return observations

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return 0

    def simulate_control(self, actions):
        Cost_EV = []
        present_cars = self.initial["present_cars"]
        P_charging = np.zeros(self.numbers_of_cars)
        grid_penalty = 0
        for car in range(self.numbers_of_cars):
            if actions[car] >= 0:
                max_charging_energy = min(
                    [
                        self.EV_param["charging_rate"],
                        (1 - self.SOC[car, self.timestep])
                        * self.EV_param["EV_capacity"],
                    ]
                )

            else:
                max_charging_energy = min(
                    [
                        self.EV_param["charging_rate"],
                        self.SOC[car, self.timestep] * self.EV_param["EV_capacity"],
                    ]
                )
            if present_cars[car, self.timestep] == 1:
                # P_charging[car] = 100 * actions[car] / 100 * max_charging_energy
                P_charging[car] = actions[car] * max_charging_energy
            else:
                P_charging[car] = 0

        for car in range(self.numbers_of_cars):
            if present_cars[car, self.timestep] == 1:
                battery = self.SOC[car, self.timestep] * self.EV_param["EV_capacity"]
                self.SOC[car, self.timestep + 1] = (
                    battery + P_charging[car]
                ) / self.EV_param["EV_capacity"]
                # self.SOC[car, self.timestep + 1] = (
                #     self.SOC[car, self.timestep]
                #     + P_charging[car] / self.EV_param["EV_capacity"]
                # )

        # Total_charging = sum(P_charging)
        total_energy_consumed = sum(P_charging)
        if total_energy_consumed > self.max_capacity:
            grid_penalty = 10 * np.exp(
                0.1 * (total_energy_consumed - self.max_capacity)
            )
            print("grid_penalty", grid_penalty)
        Grid_final = max(total_energy_consumed, 0)
        cost_1 = total_energy_consumed * self.Price[0, self.timestep]
        if self.leave:
            for i in range(len(self.leave)):
                Cost_EV.append(
                    ((1 - self.SOC[self.leave[i], self.timestep + 1]) * 2.5) ** 2
                )
                self.SOC_FINISH.append(self.SOC[self.leave[i], self.timestep + 1])

        cost_2 = sum(Cost_EV)

        discharge_pen = []
        for car in range(self.numbers_of_cars):
            if present_cars[car, self.timestep] == 1:
                if self.SOC[car, self.timestep] < 0.2:
                    penalty = ((1 - self.SOC[car, self.timestep]) * 2) ** 2
                    penalty = 2 * np.exp(5 * (0.2 - self.SOC[car, self.timestep]))
                    discharge_pen.append(penalty)
                else:
                    penalty = 0
        cost_3 = sum(discharge_pen) * 2
        cost = cost_1 + cost_2 + cost_3 + grid_penalty

        return cost, Grid_final, cost_1, cost_3

    def simulate_station(self):
        leave = []
        Arrival = self.initial["ArrivalT"]
        Departure = self.initial["DepartureT"]
        present_cars = self.initial["present_cars"]

        if self.timestep < 24:
            for car in range(self.numbers_of_cars):
                Departure_car = Departure[car]
                if present_cars[car, self.timestep] == 1 and (
                    self.timestep + 1 in Departure_car
                ):
                    leave.append(car)

        Departure_hour = []
        for car in range(self.numbers_of_cars):
            if present_cars[car, self.timestep] == 0:
                Departure_hour.append(0)
            else:
                for i in range(len(Departure[car])):
                    if self.timestep < Departure[car][i]:
                        Departure_hour.append(Departure[car][i] - self.timestep)
                        break

        Battery = []
        for car in range(self.numbers_of_cars):
            Battery.append(self.SOC[car][self.timestep])

        return leave, Departure_hour, Battery

    def Energy_calculation(self):

        minutes_of_timestep = 15

        Price_day = []

        if self.price_flag == 1:
            Price_day = np.array(
                [
                    0.25,
                    0.25,
                    0.25,
                    0.25,
                    0.25,
                    0.25,
                    0.25,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                ]
            )
        elif self.price_flag == 2:
            Price_day = np.array(
                [
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.43,
                    0.43,
                    0.43,
                    0.36,
                    0.56,
                    0.56,
                    0.56,
                    0.38,
                    0.26,
                    0.20,
                    0.20,
                    0.20,
                    0.30,
                    0.30,
                    0.30,
                    0.30,
                    0.35,
                    0.35,
                    0.35,
                ]
            )
        elif self.price_flag == 3:
            Price_day = np.array(
                [
                    0.41,
                    0.30,
                    0.26,
                    0.26,
                    0.26,
                    0.30,
                    0.30,
                    0.20,
                    0.26,
                    0.26,
                    0.46,
                    0.30,
                    0.30,
                    0.7,
                    0.7,
                    0.46,
                    0.46,
                    0.7,
                    0.52,
                    0.50,
                    0.55,
                    0.49,
                    0.56,
                    0.40,
                ]
            )
        elif self.price_flag == 4:

            Price_day = np.array(
                [
                    0.6,
                    0.6,
                    0.35,
                    0.35,
                    0.35,
                    0.35,
                    0.25,
                    0.28,
                    0.28,
                    0.31,
                    0.31,
                    0.31,
                    0.31,
                    0.31,
                    0.31,
                    0.31,
                    0.31,
                    0.16,
                    0.16,
                    0.16,
                    0.21,
                    0.21,
                    0.21,
                    0.21,
                ]
            )

        Price_day = np.concatenate([Price_day, Price_day], axis=0)
        Price = np.zeros((self.number_of_days, 48))
        for i in range(0, self.number_of_days):
            Price[i, :] = Price_day

        return Price
