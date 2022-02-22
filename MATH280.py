print('MATH280')

def rkf45(symfun,depvars,inits,t_range):
    from scipy.integrate import solve_ivp
    from sympy import lambdify
    if isinstance(depvars,list): depvars=tuple(depvars)
    fun = lambdify((t_range[0],depvars),symfun)
    if not isinstance(inits,list): inits=[inits]
    ns = solve_ivp(fun,[t_range[1],t_range[2]],inits,method='RK45')
    return [ns.t, ns.y]

def BDF(symfun,depvars,inits,t_range):
    from scipy.integrate import solve_ivp
    from sympy import lambdify
    if isinstance(depvars,list): depvars=tuple(depvars)
    fun = lambdify((t_range[0],depvars),symfun)
    if not isinstance(inits,list): inits=[inits]
    ns = solve_ivp(fun,[t_range[1],t_range[2]],inits,method='BDF')
    return [ns.t, ns.y]

def drawdf(rhs_sym_expression,xlist,ylist, **kwargs):
    from sympy import lambdify
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    '''
    drawdf(x+y,[x,-1,1],[y,-1,1],soln_at=[[.1,.2],[-.1,-.2]],plot_title="Title")
    
    drawdf([v,-x],[x,-1,1],[v,-1,1],soln_at=[[.1,.2],[-.5,-.2]],plot_title="Title")
    
    '''
    rhsn = lambdify((xlist[0],ylist[0]),rhs_sym_expression,'numpy')
# Vector field
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(xlist[1], xlist[2], 20), np.linspace(ylist[1], ylist[2], 20))
    if isinstance(rhs_sym_expression,list):
        U =  rhsn(X,Y)[0]
        V = rhsn(X,Y)[1]
    else:
        U = 1.0
        V = rhsn(X,Y)
    # Normalize arrows
    N = np.sqrt(U ** 2 + V ** 2)
    U = U / N
    V = V / N
    ax.quiver(X, Y, U, V, angles="xy")


    #  Solution curves
    init_points=kwargs.get("soln_at",None)    
    if init_points != None: # Are initial points given?
        if not isinstance(init_points[1],list): init_points = [ init_points ]  
        #Construct the RHS function for the numerical solver
        if isinstance(rhs_sym_expression,list): # Vector rhs function: Phase Space
              def vf(t,xx):
                dx = np.zeros(2)
                dx[0] = rhsn(xx[0],xx[1])[0]
                dx[1] = rhsn(xx[0],xx[1])[1]
                return dx 
        else:     # 1-Dimension rhs function
              def vf(t,xx):
                dx = np.zeros(2)
                dx[0] = 1.0
                dx[1] = rhsn(xx[0],xx[1]) 
                return dx 
            
        # Now compute the solutions and plot
        def too_big(t,y): # set  a stopping event if soln gets too big
#            buf = .1
#            return (ylist[2]+buf-y[1])*(y[1]-(ylist[1]-buf))
            return (100-y[1])*(y[1]-100)        
        too_big.terminal=True         
        for y_initial in init_points: 
              t0 = y_initial[0]
              tEnd = 20.0
              yy = solve_ivp(vf,  [t0, tEnd], y_initial, events=too_big, max_step=0.1)
              plt.plot(yy.y[0], yy.y[1], "-")

    
#clean up the axes
    titlestring = kwargs.get("plot_title",None)
    ax.set_title(titlestring)
    ax.set_xlim([xlist[1], xlist[2]])
    ax.set_ylim([ylist[1], ylist[2]])
    ax.set_xlabel(str(xlist[0]))
    ax.set_ylabel(str(ylist[0]))
    
    return ax
	
def laplace(ee,t,s):
    '''
    A hack of sympy built-in laplace_transform() to properly treat 
      L(f'(t))=sL(f(t))-f(0)
      and
      L(f''(t))=s^2L(f(t))-sf(0)-f'(0)
      
      The idea comes from code posted in  github discussion thread
      https://github.com/sympy/sympy/issues/7219
      
      
      Here's an example calling sequence to solve an ode
      x'' + x = delta_{pi/2}, x(0)=0
      
      eq4 = Eq(x(t).diff(t,2)+x(t),DiracDelta(t-pi/2))
      L=laplace(eq4,t,s)
      L0=L.subs(x(0),1).subs(Subs(Derivative(x(t), t), t, 0),0)
      Lx=solve(L0,LaplaceTransform(x(t),t,s))
      laplaceInv(Lx,s,t)
      
    '''
    import sympy

    if isinstance(ee,sympy.Equality): return sympy.Eq(laplace(ee.lhs,t,s),laplace(ee.rhs,t,s))
    bigLS = 0
    #print('line 13', isinstance(ee,sympy.Add) )
    if isinstance(ee,sympy.Add):
        for ff in ee.args:
            #print('line 16', ff)
            ls=sympy.laplace_transform(ff,t,s,noconds=True)   
            #print('line 18', ls, ls.atoms())
            if t in ls.atoms():
                #print('line 20', ls.args)
                i0=0
                if isinstance(ls,sympy.Mul): 
                    i0=1  #skip the first term...it's a constant
                    #print('line 24', i0)
                if i0==0:
                    dterm = ls.args[i0]
                elif i0==1:
                    dterm = ls.args[i0].args[0]
                if isinstance(dterm,sympy.Derivative):
                    #print(ls.args[i0].args[0])
                    f =  dterm.args[0]
                    #print('line 79',f)
                    n = dterm.args[1][1]
                    #print('line 29', n)
                    #print('line 30', f)
                    rexp=s*sympy.LaplaceTransform(f,t,s) - f.subs(t,0)            
                    #print('line 32',rexp)
                    if n==1: lsout=ls.replace(sympy.LaplaceTransform(sympy.Derivative(f,t),t,s),rexp)
                    if n==2: 
                       rexp = s*rexp - sympy.Subs(sympy.Derivative(f,t),t,0) 
                       #print('line 36',rexp)
                       lsout=ls.replace(sympy.LaplaceTransform(sympy.Derivative(f,t,2),t,s),rexp)
                    bigLS = bigLS + lsout       
                    #print('line 39', bigLS)
                else:
                    bigLS= bigLS + ls
            else:
                bigLS = bigLS + ls
                #print('line 44',bigLS)
    else:  #a single term
        ls=sympy.laplace_transform(ee,t,s,noconds=True)   
        #print(ls, t in ls.atoms())
        if t in ls.atoms():
            #print('line 70', ls.args)
            i0=0
            if isinstance(ls,sympy.Mul): i0=1  #skip the first term...it's a constant
            #print('line 73', i0)
            #print('line 74', ls.args[i0])
            if i0==0:
                dterm = ls.args[i0]
            elif i0==1:
                dterm = ls.args[i0].args[0]
            #if isinstance(ls.args[i0].args[0],sympy.Derivative):
            if isinstance(dterm,sympy.Derivative):
                #print(ls.args[i0].args[0])
                f =  dterm.args[0]
                #print('line 79',f)
                n = dterm.args[1][1]
                #print(f)
                rexp=s*sympy.LaplaceTransform(f,t,s) - f.subs(t,0)            
                if n==1: lsout=ls.replace(sympy.LaplaceTransform(sympy.Derivative(f,t),t,s),rexp)
                if n==2: 
                   rexp = s*rexp - sympy.Subs(sympy.Derivative(f,t),t,0) 
                   lsout=ls.replace(sympy.LaplaceTransform(sympy.Derivative(f,t,2),t,s),rexp)
                bigLS = bigLS + lsout       
            else:
                bigLS=bigLS+ls
        else: bigLS = bigLS + ls
    return sympy.expand(bigLS)



def laplaceInv(e,s,t):
    '''
    a tiny hack of inverse_laplace_transform to extract 0th element of list input
    
    e is presumed to be the output of laplace
    
      Here's an example calling sequence to solve an ode
      x'' + x = delta_{pi/2}, x(0)=0
      
      eq4 = Eq(x(t).diff(t,2)+x(t),DiracDelta(t-pi/2))
      L=laplace(eq4,t,s)
      L0=L.subs(x(0),1).subs(Subs(Derivative(x(t), t), t, 0),0)
      Lx=solve(L0,LaplaceTransform(x(t),t,s))
      laplaceInv(Lx,s,t)

    '''
    import sympy
    if isinstance(e,list): e=e[0] # a preceding call to solve returns a one-element vector
    return sympy.expand(sympy.inverse_laplace_transform(e,s,t))