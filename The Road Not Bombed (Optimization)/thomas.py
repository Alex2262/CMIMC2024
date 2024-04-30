import random


class Planner:
    def setup(self, pairs, bd):
        self.n = 16
        self.pairs = pairs
        self.bd = bd
        self.bestPlan = [[1] * self.n for i in range(self.n)]
        self.sentPlan = []
        return

    def sus_random(self, q, queryOutputs):  # p = 5, bd = 0.25
        l = len(queryOutputs)
        if l > 0 and queryOutputs[l - 1]:
            self.bestPlan = self.sentPlan
        n = self.n
        newPlan = [[1] * n for i in range(n)]

        # take the most recent successful plan
        # remove each road with small probability
        done = False
        while not done:
            rando = int(random.uniform(0, 256))
            if newPlan[rando//16][rando%16] == 0 or ((rando/16 == coord[0] and rando%16 == coord[1]) for coord in pair3 for pair3 in self.pairs):
                self.sentPlan = newPlan
        return self.sentPlan

    def task2(self, q, queryOutputs):
        return self.task1(q, queryOutputs)

    def task3(self, q, queryOutputs):
        return self.task1(q, queryOutputs)

    def task4(self, q, queryOutputs):
        return self.task1(q, queryOutputs)

    def query(self, q, queryOutputs):
        # feel free to modify this function, this is just a suggestion
        if len(self.pairs) == 5 and self.bd == 0.25:
            return self.sus_random(q, queryOutputs)
        
        if len(self.pairs) == 5 and self.bd == 0.1:
            return self.sus_random(q, queryOutputs)
        
        if len(self.pairs) == 1 and self.bd == 0.25:
            return self.sus_random(q, queryOutputs)
        
        if len(self.pairs) == 1 and self.bd == 0.1:
            return self.sus_random(q, queryOutputs)