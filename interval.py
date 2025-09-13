class Interval:
    def __init__(self, min_val=float('inf'), max_val=-float('inf')):
        self.min = min_val
        self.max = max_val
    
    def contains(self, x):
        return self.min <= x <= self.max
    
    def surrounds(self, x):
        return self.min < x < self.max
    
    def clamp(self, x):
        if x < self.min:
            return self.min
        if x > self.max:
            return self.max
        return x