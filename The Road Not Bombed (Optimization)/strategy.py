import random


class Planner:
    def setup(self, pairs, bd):
        self.n = 16
        self.pairs = pairs
        self.bd = bd
        self.yeetProbability = 0.07  # probability of removing a road
        self.bestPlan = [[1] * self.n for i in range(self.n)]
        self.sentPlan = []
        self.t4Recent = [[-2020, -2020]]
        self.t4Blacklist = []
        self.found = 0
        self.t4Phase2 = 0
        self.recentFlood = 0
        return
    def count(self, xd):
        n = self.n
        ans = 0
        for i in range(n):
            for j in range(n):
                if (xd[i][j] != 0):
                    ans += 1
        return ans


    def task1(self, q, queryOutputs):  # p = 5, bd = 0.25
        # print(str(self.found) + "debugger22")
        l = len(queryOutputs)
        if l > 0 and queryOutputs[l - 1]:
            if self.found == 0:
                self.found = 1
            if (self.count(self.sentPlan) < self.count(self.bestPlan)):
                #if (self.found == 0 or self.found == 1): # for printing only
                    #print("best")
                    #for i in self.bestPlan:
                    #    print(i)
                    #print("next")
                    #for i in self.sentPlan:
                    #    print(i)
                    #print("nexus")

                self.bestPlan = self.sentPlan
                #print("plan best change " + str(q) + str(self.found))
                
        
        if (l > 0 and queryOutputs[-1] == 0):
            
            if (self.found == 2):
                #print("ACTIVATING BLACKLIST?")
                if ([self.t4Recent[0][0], self.t4Recent[0][1]] not in self.t4Blacklist and not (self.t4Recent[0][0] == -2020 or self.t4Recent[0][1] == -2020)):
                    self.t4Blacklist.append([-1,-1])
                    self.t4Blacklist[-1][0] = self.t4Recent[0][0]
                    self.t4Blacklist[-1][1] = self.t4Recent[0][1]
                    #print(self.t4Blacklist)
        # elif not (self.t4Recent[0][0] == -2020 or self.t4Recent[0][1] == -2020):
            
            
        n = self.n
        newPlan = []
        for i in range(n):
            newPlan.append([])
            for j in range(n):
                newPlan[-1].append(0)
        for i in range(n):
            for j in range(n):
                newPlan[i][j] = self.bestPlan[i][j]
        #for i in self.bestPlan:
            #if (q > 40):

                # print(i)
        #print("$$$$$$$$$$$")
        if (q >= 33 or self.found == 0):
            #print("randomizer" + str(q))
            # take the most recent successful plan
            # remove each road with small probability
            for i in range(n):
                for j in range(n):
                    newPlan[i][j] = self.bestPlan[i][j]
                    if newPlan[i][j] == 1:
                        rando = random.uniform(0, 1)
                        if rando < 0 * self.yeetProbability + 0.075:
                            newPlan[i][j] = 0
            marks = []
            for i in range(n):
                for j in range(n):
                    if (0 < i and i < n-1 and 0 < j and j < n-1):
                        neighbors = int((newPlan[i+1][j] == 1) + (newPlan[i][j+1] == 1) + (newPlan[i-1][j] == 1) + (newPlan[i][j-1] == 1))
                        if (neighbors <= 1):
                            marks.append([i,j])
            for a in marks:
                pass
                #newPlan[a[0]][a[1]] = 0

            


            self.sentPlan = newPlan
        
            return self.sentPlan
        elif (q < 33 and self.found == 1 and self.t4Phase2 == 0):
            self.t4Phase2 = 1
            self.sentPlan = self.bestPlan
            return self.sentPlan
        elif (self.t4Phase2 == 1):
            
            self.found = 2
            #print(str(self.found) + "oeoeoeoeoeo")
            self.t4Phase2 = 2
            #print("poop")
            

            # begin flood fill
            #print("Flood fill!")
            # newPlan = []
            for i in range(n):
                for j in range(n):
                    newPlan[i][j] = self.bestPlan[i][j]
                # print(newPlan[i])
            visited = []
            vcount = 0
            for i in range(n):
                visited.append([])
                for j in range(n):
                    visited[-1].append(0)
            for i in range(n):
                for j in range(n):
                    if (newPlan[i][j] == 0):
                        visited[i][j] = 1
                        vcount += 1
            ti = 0
            tj = 0
            while (vcount < 256):
                if (visited[ti][tj] == 1):
                    ti += 1
                    if (ti == n):
                        ti = 0
                        tj += 1
                    continue
                queue = []
                group = []
                queue.append([ti, tj])
                visited[ti][tj] = 1
                while (len(queue) > 0):
                    tx, ty = queue[0][0], queue[0][1]
                    group.append([tx, ty])
                    if (tx > 0):
                        if (visited[tx-1][ty] == 0):
                            queue.append([tx-1, ty])
                            visited[tx-1][ty] = 1
                        
                    if (ty > 0):
                        if (visited[tx][ty-1] == 0):
                            queue.append([tx, ty-1])
                            visited[tx][ty-1] = 1
                    if (tx < n-1):
                        if (visited[tx+1][ty] == 0):
                            queue.append([tx+1, ty])
                            visited[tx+1][ty] = 1
                    if (ty < n-1):
                        if (visited[tx][ty+1] == 0):
                            queue.append([tx, ty+1])
                            visited[tx][ty+1] = 1
                    queue.pop(0)
                    vcount += 1
                
                onEdge = 0
                for ae in group:
                    if (ae[0] == 0 or ae[0] == n-1 or ae[1] == 0 or ae[1] == n-1):
                        onEdge += 1
                        if (onEdge > 1):
                            break
                if (onEdge <= 1):
                    for ae in group:
                        newPlan[ae[0]][ae[1]] = 0
            #print("erer")
            #for iei in newPlan:
                #print(iei)
            self.recentFlood = 0
            self.sentPlan = newPlan
        
            return self.sentPlan

        elif (self.found == 2):
            marks = []
            kc = 0
            for i in range(n):
                for j in range(n):
                    if (0 < i and i < n-1 and 0 < j and j < n-1):
                        neighbors = 0
                        if (newPlan[i+1][j] == 1):
                            neighbors += 1
                        if (newPlan[i][j+1] == 1):
                            neighbors += 1
                        if (newPlan[i-1][j] == 1):
                            neighbors += 1
                        if (newPlan[i][j-1] == 1):
                            neighbors += 1
                        
                        if (neighbors <= 1):
                            newPlan[i][j] = 0
                            kc += 1
            for a in marks:
                pass
                #newPlan[a[0]][a[1]] = 0
            # if (kc >= 5):
            #     for a in newPlan:
            #         print(a)
            #         print("--3---33---\n")


            #print(str(self.found) + "debugger")
            if self.recentFlood == 5 and queryOutputs[-1] == True:
                #begin floodfilling
                for i in range(n):
                    for j in range(n):
                        newPlan[i][j] = self.bestPlan[i][j]
                    # print(newPlan[i])
                visited = []
                vcount = 0
                for i in range(n):
                    visited.append([])
                    for j in range(n):
                        visited[-1].append(0)
                for i in range(n):
                    for j in range(n):
                        if (newPlan[i][j] == 0):
                            visited[i][j] = 1
                            vcount += 1
                ti = 0
                tj = 0
                while (vcount < 256):
                    if (visited[ti][tj] == 1):
                        ti += 1
                        if (ti == n):
                            ti = 0
                            tj += 1
                        continue
                    queue = []
                    group = []
                    queue.append([ti, tj])
                    visited[ti][tj] = 1
                    while (len(queue) > 0):
                        tx, ty = queue[0][0], queue[0][1]
                        group.append([tx, ty])
                        if (tx > 0):
                            if (visited[tx-1][ty] == 0):
                                queue.append([tx-1, ty])
                                visited[tx-1][ty] = 1
                            
                        if (ty > 0):
                            if (visited[tx][ty-1] == 0):
                                queue.append([tx, ty-1])
                                visited[tx][ty-1] = 1
                        if (tx < n-1):
                            if (visited[tx+1][ty] == 0):
                                queue.append([tx+1, ty])
                                visited[tx+1][ty] = 1
                        if (ty < n-1):
                            if (visited[tx][ty+1] == 0):
                                queue.append([tx, ty+1])
                                visited[tx][ty+1] = 1
                        queue.pop(0)
                        vcount += 1
                    
                    onEdge = 0
                    for ae in group:
                        if (ae[0] == 0 or ae[0] == n-1 or ae[1] == 0 or ae[1] == n-1):
                            onEdge += 1
                            if (onEdge > 1):
                                break
                    if (onEdge <= 1):
                        for ae in group:
                            pass
                            newPlan[ae[0]][ae[1]] = 0
                # end floodfilling
                self.recentFlood = 0
                self.sentPlan = newPlan
        
                return self.sentPlan
            


            if (queryOutputs[-1] == True):
                self.recentFlood += 1
            done = False #finding non blacklisted road
            
            

            for i in range(n-1, -1, -1):
                
                for j in range(n-1, -1, -1):
                    
                    if newPlan[i][j] == 1 and not done:
                        if ([i, j] in self.t4Blacklist):
                            # print("yyee")
                            continue
                        else:

                            self.t4Recent[0][0] = i
                            self.t4Recent[0][1] = j
                            newPlan[i][j] = 0
                            done = True
                            break
                    if (done):
                        break
                if (done):
                    break
            
            self.sentPlan = newPlan

            return self.sentPlan
        else:
            #print(self.found)
            #print("reeee")
            #newPlan[16][16] = 0
            self.sentPlan = newPlan
        
            return self.sentPlan
        

        self.sentPlan = newPlan
        
        return self.sentPlan

    def task2(self, q, queryOutputs):
        # return self.task1(q, queryOutputs)
        l = len(queryOutputs)

        if l > 0 and queryOutputs[l - 1]:
            self.bestPlan = self.sentPlan
        n = self.n
        newPlan = [[22] * n for i in range(n)]

        # take the most recent successful plan
        # remove each road with small probability
        for i in range(n):
            for j in range(n):
                newPlan[i][j] = self.bestPlan[i][j]
                if newPlan[i][j] == 1:
                    rando = random.uniform(0, 1)
                    if rando < 0.06:
                        newPlan[i][j] = 0

        self.sentPlan = newPlan
        return self.sentPlan

    def task3(self, q, queryOutputs):
        return self.task1(q, queryOutputs)
        
    def task4(self, q, queryOutputs): # p = 1 bd = 0.1
        # print(str(self.found) + "debugger22")
        l = len(queryOutputs)
        if l > 0 and queryOutputs[l - 1]:
            if self.found == 0:
                self.found = 1
            if (self.count(self.sentPlan) < self.count(self.bestPlan)):
                #if (self.found == 0 or self.found == 1): # for printing only
                    #print("best")
                    #for i in self.bestPlan:
                    #    print(i)
                    #print("next")
                    #for i in self.sentPlan:
                    #    print(i)
                    #print("nexus")

                self.bestPlan = self.sentPlan
                #print("plan best change " + str(q) + str(self.found))
                
        
        if (l > 0 and queryOutputs[-1] == 0):
            
            if (self.found == 2):
                #print("ACTIVATING BLACKLIST?")
                if ([self.t4Recent[0][0], self.t4Recent[0][1]] not in self.t4Blacklist and not (self.t4Recent[0][0] == -2020 or self.t4Recent[0][1] == -2020)):
                    self.t4Blacklist.append([-1,-1])
                    self.t4Blacklist[-1][0] = self.t4Recent[0][0]
                    self.t4Blacklist[-1][1] = self.t4Recent[0][1]
                    #print(self.t4Blacklist)
        # elif not (self.t4Recent[0][0] == -2020 or self.t4Recent[0][1] == -2020):
            
            
        n = self.n
        newPlan = [[-20] * n for i in range(n)]
        #for i in self.bestPlan:
            #if (q > 40):

                # print(i)
        #print("$$$$$$$$$$$")
        if (q >= 33 or self.found == 0):
            #print("randomizer" + str(q))
            # take the most recent successful plan
            # remove each road with small probability
            for i in range(n):
                for j in range(n):
                    newPlan[i][j] = self.bestPlan[i][j]
                    if newPlan[i][j] == 1:
                        rando = random.uniform(0, 1)
                        if rando < 0 * self.yeetProbability + 0.075:
                            newPlan[i][j] = 0
            self.sentPlan = newPlan
        
            return self.sentPlan
        elif (q < 33 and self.found == 1 and self.t4Phase2 == 0):
            self.t4Phase2 = 1
            self.sentPlan = self.bestPlan
            return self.sentPlan
        elif (self.t4Phase2 == 1):
            
            self.found = 2
            #print(str(self.found) + "oeoeoeoeoeo")
            self.t4Phase2 = 2
            #print("poop")
            

            # begin flood fill
            #print("Flood fill!")
            # newPlan = []
            for i in range(n):
                for j in range(n):
                    newPlan[i][j] = self.bestPlan[i][j]
                # print(newPlan[i])
            visited = []
            vcount = 0
            for i in range(n):
                visited.append([])
                for j in range(n):
                    visited[-1].append(0)
            for i in range(n):
                for j in range(n):
                    if (newPlan[i][j] == 0):
                        visited[i][j] = 1
                        vcount += 1
            ti = 0
            tj = 0
            while (vcount < 256):
                if (visited[ti][tj] == 1):
                    ti += 1
                    if (ti == n):
                        ti = 0
                        tj += 1
                    continue
                queue = []
                group = []
                queue.append([ti, tj])
                visited[ti][tj] = 1
                while (len(queue) > 0):
                    tx, ty = queue[0][0], queue[0][1]
                    group.append([tx, ty])
                    if (tx > 0):
                        if (visited[tx-1][ty] == 0):
                            queue.append([tx-1, ty])
                            visited[tx-1][ty] = 1
                        
                    if (ty > 0):
                        if (visited[tx][ty-1] == 0):
                            queue.append([tx, ty-1])
                            visited[tx][ty-1] = 1
                    if (tx < n-1):
                        if (visited[tx+1][ty] == 0):
                            queue.append([tx+1, ty])
                            visited[tx+1][ty] = 1
                    if (ty < n-1):
                        if (visited[tx][ty+1] == 0):
                            queue.append([tx, ty+1])
                            visited[tx][ty+1] = 1
                    queue.pop(0)
                    vcount += 1
                
                onEdge = 0
                for ae in group:
                    if (ae[0] == 0 or ae[0] == n-1 or ae[1] == 0 or ae[1] == n-1):
                        onEdge += 1
                        if (onEdge > 1):
                            break
                if (onEdge <= 1):
                    for ae in group:
                        newPlan[ae[0]][ae[1]] = 0
            #print("erer")
            #for iei in newPlan:
                #print(iei)
            self.recentFlood = 0
            self.sentPlan = newPlan
        
            return self.sentPlan

        elif (self.found == 2):
            #print(str(self.found) + "debugger")
            if self.recentFlood == 5 and queryOutputs[-1] == True:
                #begin floodfilling
                for i in range(n):
                    for j in range(n):
                        newPlan[i][j] = self.bestPlan[i][j]
                    # print(newPlan[i])
                visited = []
                vcount = 0
                for i in range(n):
                    visited.append([])
                    for j in range(n):
                        visited[-1].append(0)
                for i in range(n):
                    for j in range(n):
                        if (newPlan[i][j] == 0):
                            visited[i][j] = 1
                            vcount += 1
                ti = 0
                tj = 0
                while (vcount < 256):
                    if (visited[ti][tj] == 1):
                        ti += 1
                        if (ti == n):
                            ti = 0
                            tj += 1
                        continue
                    queue = []
                    group = []
                    queue.append([ti, tj])
                    visited[ti][tj] = 1
                    while (len(queue) > 0):
                        tx, ty = queue[0][0], queue[0][1]
                        group.append([tx, ty])
                        if (tx > 0):
                            if (visited[tx-1][ty] == 0):
                                queue.append([tx-1, ty])
                                visited[tx-1][ty] = 1
                            
                        if (ty > 0):
                            if (visited[tx][ty-1] == 0):
                                queue.append([tx, ty-1])
                                visited[tx][ty-1] = 1
                        if (tx < n-1):
                            if (visited[tx+1][ty] == 0):
                                queue.append([tx+1, ty])
                                visited[tx+1][ty] = 1
                        if (ty < n-1):
                            if (visited[tx][ty+1] == 0):
                                queue.append([tx, ty+1])
                                visited[tx][ty+1] = 1
                        queue.pop(0)
                        vcount += 1
                    
                    onEdge = 0
                    for ae in group:
                        if (ae[0] == 0 or ae[0] == n-1 or ae[1] == 0 or ae[1] == n-1):
                            onEdge += 1
                            if (onEdge > 1):
                                break
                    if (onEdge <= 1):
                        for ae in group:
                            pass
                            newPlan[ae[0]][ae[1]] = 0
                # end floodfilling
                self.recentFlood = 0
                self.sentPlan = newPlan
        
                return self.sentPlan
            


            if (queryOutputs[-1] == True):
                self.recentFlood += 1
            done = False

            for i in range(n-1, -1, -1):
                
                for j in range(n-1, -1, -1):
                    
                    if newPlan[i][j] == 1 and not done:
                        if ([i, j] in self.t4Blacklist):
                            # print("yyee")
                            continue
                        else:

                            self.t4Recent[0][0] = i
                            self.t4Recent[0][1] = j
                            newPlan[i][j] = 0
                            done = True
                            break
                    if (done):
                        break
                if (done):
                    break
            
            self.sentPlan = newPlan

            return self.sentPlan
        else:
            #print(self.found)
            #print("reeee")
            #newPlan[16][16] = 0
            self.sentPlan = newPlan
        
            return self.sentPlan
        

        self.sentPlan = newPlan
        
        return self.sentPlan
        

    def query(self, q, queryOutputs):
        # feel free to modify this function, this is just a suggestion
        if len(self.pairs) == 5 and self.bd == 0.25:
            return self.task1(q, queryOutputs)
        
        if len(self.pairs) == 5 and self.bd == 0.1:
            return self.task1(q, queryOutputs)
        
        if len(self.pairs) == 1 and self.bd == 0.25:
            return self.task1(q, queryOutputs)
        
        if len(self.pairs) == 1 and self.bd == 0.1:
            return self.task1(q, queryOutputs)
        
