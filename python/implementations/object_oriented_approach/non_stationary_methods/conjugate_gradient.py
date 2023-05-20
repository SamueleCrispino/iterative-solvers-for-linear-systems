from implementations.object_oriented_approach.non_stationary_methods.gradient import Gradient

class Conjugate_gradient(Gradient):
    
    def __init__(self, tol, a, b, real_x):
        super().__init__(tol, a, b, real_x)
        self.method_name = "CONJUGATE_GRADIENT_METHOD"
        self.r_next = None
        self.d_next = None
        self.w = None
        self.beta = None 
        self.r = None
        self.d = None
        self.z = None

    def compute_residue(self):
        if self.k == 0:
            super().compute_residue()
        else:
            self.r = self.r_next

    def compute_y(self):
        self.y = self.a.dot(self.d)

    def compute_gradient_alfa(self):
        if self.k == 0:
            self.d = self.r
        else:
            self.d = self.d_next
        self.compute_y()
        self.z = self.d.dot(self.y)
        self.alfa = self.d.dot(self.r) / self.z

    def compute_next_x(self):

        self.compute_gradient_alfa()
        self.x = self.x - self.alfa*self.d

        self.r_next = self.a.dot(self.x) - self.b
        self.w = self.a.dot(self.r_next)

        self.beta = self.d.dot(self.w) / self.z
        
        # + / - 
        self.d_next = self.r_next - self.beta*self.d




    

    