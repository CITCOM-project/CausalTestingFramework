from causal_testing.testing.causal_test_case import CausalTestCase
import numpy as np
import pandas as pd


class FairHeartRateDrugTrial:
    """ A simple model which simulates the effect of a drug on heart rate. """

    def __init__(self, age, weight, heart_rate):
        self.age = age
        self.weight = weight
        self.heart_rate = heart_rate
        self.drug = 0
        self.take_drug()

    def take_drug(self):
        if np.random.randint(0, 2) > 0:  # treatment is assigned at random
            self.drug = 1
            self.heart_rate += 20  # additive effect for all
            if self.age > 60:
                self.heart_rate *= (1 + self.age/200)  # small multiplicative effect for those over 60 years old
            if self.weight > 90:
                self.heart_rate *= (1 + self.weight/200)  # small multiplicative effect for those over 90kg


class ConfoundedHeartRateDrugTrial:
    """ A simple model which simulates the effect of a drug on heart rate. """

    def __init__(self, age, weight, heart_rate):
        self.age = age
        self.weight = weight
        self.heart_rate = heart_rate
        self.drug = 0
        self.take_drug()

    def take_drug(self):
        treatment_prob = np.random.random() + (self.weight/200)  # treatment is assigned based on weight
        if treatment_prob > .5:
            self.drug = 1
            self.heart_rate += 20  # additive effect for all
            if self.age > 60:
                self.heart_rate *= (1 + self.age/200)  # small multiplicative effect for those over 60 years old
            if self.weight > 90:
                self.heart_rate *= (1 + self.weight/200)  # small multiplicative effect for those over 90kg


class TestFairDrugTrial(CausalTestCase):

    def collect_data(self):
        fair_df = pd.DataFrame({"drug": [], "hr": [], "age": [], "weight": []})

        for n in range(1000):
            age = abs(np.random.normal(40, 10))
            weight = abs(np.random.normal(70, 10))
            heart_rate = abs(np.random.normal(60, 5))
            fair_experiment = FairHeartRateDrugTrial(age, weight, heart_rate)
            fair_post_treatment_heart_rate = fair_experiment.heart_rate
            fair_drug = fair_experiment.drug
            fair_df = fair_df.append({"drug": fair_drug, "hr": fair_post_treatment_heart_rate, "age": age,
                                      "weight": weight}, ignore_index=True)

    def apply_intervention(self):
        pass

    def estimate_causal_effect(self):
        pass


class ConfoundedDrugTrial(CausalTestCase):

    def collect_data(self):
        confounded_df = pd.DataFrame({"drug": [], "hr": [], "age": [], "weight": []})

        for n in range(1000):
            age = abs(np.random.normal(40, 10))
            weight = abs(np.random.normal(70, 10))
            heart_rate = abs(np.random.normal(60, 5))
            confounded_experiment = ConfoundedHeartRateDrugTrial(age, weight, heart_rate)
            confounded_post_treatment_heart_rate = confounded_experiment.heart_rate
            confounded_drug = confounded_experiment.drug
            confounded_df.append({"drug": confounded_drug, "hr": confounded_post_treatment_heart_rate,
                                  "age": age, "weight": weight}, ignore_index=True)

    def apply_intervention(self):
        pass

    def estimate_causal_effect(self):
        pass


if __name__ == "__main__":
    fair_causal_test_case = FairHeartRateDrugTrial
    confounded_causal_test_case = ConfoundedDrugTrial
