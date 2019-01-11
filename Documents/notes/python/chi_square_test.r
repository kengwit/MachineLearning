#
# Simulate data, one iteration per column of `x`.
#
n <- 20
n.sim <- 1e4
bins <- qnorm(seq(0, 1, 1/4))
x <- matrix(rnorm(n*n.sim), nrow=n)
#
# Compute statistics.
#

# compute mean of each column
m <- colMeans(x)

# compute std dev of each column
s <- apply(sweep(x, 2, m), 2, sd)

# create "count" where
# rows are bins 1,2,3,4
# cols are the 10,000 simulations
counts <- apply(matrix(as.numeric(cut(x, bins)), nrow=n), 2, tabulate, nbins=4)

#   for each m[i], s[i] pair, i = 1,...,n.sim
#      generate pdf[i] = pnorm(bins, m[i], s[i]) corresponding to the vector of percentiles given by "bins"
#      calculate diff( pdf[i] ) i.e. calculate the increment sizes in the "Y-axis"
expectations <- mapply(function(m,s) n*diff(pnorm(bins, m, s)), m, s)

# colSums sums each column
chisquared <- colSums((counts - expectations)^2 / expectations)
#
# Plot histograms of means, variances, and chi-squared stats.  The first
# two confirm all is working as expected.
#
mfrow <- par("mfrow")
par(mfrow=c(1,3))
red <- "#a04040"  # Intended to show correct distributions
blue <- "#404090" # To show the putative chi-squared distribution
hist(m, freq=FALSE)
curve(dnorm(x, sd=1/sqrt(n)), add=TRUE, col=red, lwd=2)
hist(s^2, freq=FALSE)
curve(dchisq(x*(n-1), df=n-1)*(n-1), add=TRUE, col=red, lwd=2)
hist(chisquared, freq=FALSE, breaks=seq(0, ceiling(max(chisquared)), 1/4), 
     xlim=c(0, 13), ylim=c(0, 0.55), 
     col="#c0c0ff", border="#404040")
curve(ifelse(x <= 0, Inf, dchisq(x, df=2)), add=TRUE, col=red, lwd=2)
curve(ifelse(x <= 0, Inf, dchisq(x, df=1)), add=TRUE, col=blue, lwd=2)
par(mfrow=mfrow)