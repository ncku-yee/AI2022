class Solution:
    def __init__(self):
        line = raw_input().split(' ')       # Readline and split
        self.N = int(line[0])               # N
        self.M = int(line[1])               # M
        self.bridges = []                   # Records the bridges
        for i in range(self.M):
            line = raw_input().split(' ')   # Readline and split
            self.bridges.append((int(line[0]), int(line[1])))
    #Feel free to define your own member function
    
    def solve(self):
        candidates = [1]                    # Default candidates is 1.
        path = []                           # Path that Capybaras go through. 
        while(candidates):                  # Herd Capybaras until there is no candidates.
            current = min(candidates)       # Get the candidate in candidates list.
            candidates = []                 # Clear the candidates list.
            path.append(str(current))       # Add candidate to the path.
            for bridge in self.bridges:
                if(bridge[0] == current and bridge[1] > current):   # Next candidate must larger than current island.
                    candidates.append(bridge[1])
        print(" ".join(path))

if __name__ == '__main__':
    ans = Solution()
    ans.solve()