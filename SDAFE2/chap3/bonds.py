###########  R script for Chapter 3   ####################################################
###########  of Statistics and Data Analysis for Financial Engineering, 2nd Edition ######
###########  by Ruppert and Matteson  ####################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize

################################################################
#####  Code for section 3.4 and figure 3.1  ####################
################################################################
def bondvalue(c,T,r,par): 

  #
  #   Computes bv = bond values (current prices) corresponding
  #       to all values of yield to maturity in the
  #       input vector r
  #
  #       INPUT
  #        c = coupon payment (semi-annual)
  #        T = time to maturity (in years)
  #        r = vector of yields to maturity (semi-annual rates)
  #        par = par value
  #
  bv = c/r + (par - c/r) * (1+r)**(-2*T)
  return bv


#   Computes the yield to maturity of a bond paying semi-annual
#   coupon payments
#
#   price, coupon payment, and time to maturity (in years)
#   are set below
#
#   Uses the function "bondvalue"
#
price = 1200    #   current price of the bond
C = 40          #   coupon payment
T= 30           #   time to maturity
par = 1000      #   par value of the bond


yield_to_mat = np.linspace(.02,.05,num=300)
bond_price = bondvalue(C,T,yield_to_mat,par) 
plt.plot(yield_to_mat,bond_price)
plt.xlabel('yield to maturity')
plt.ylabel('price of bond')

#yield2M = spline(value,r,xout=price) 

#tck = interpolate.splrep(value,r) # note: we want to interpolate x and a function of y
#r_price = interpolate.splev(price, tck)


# create the interpolated function, and then the offset
# function used to find the roots

interp_fn = interpolate.interp1d(yield_to_mat, bond_price, 'cubic')
#plt.plot(interp_fn.x,interp_fn.y,'o')
interp_fn2 = lambda x: interp_fn(x)-price
initial_guess = 0.02
root = optimize.newton(interp_fn2,initial_guess)
plt.plot([root,root],[np.min(bond_price),np.max(bond_price)],'r-')
plt.plot([np.min(yield_to_mat),np.max(yield_to_mat)],[price,price],'r-')
#interp_fn2 = lambda x: interp_fn(x) - price