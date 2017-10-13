## The Heston Stochastic Volatility model
##
## - Closed form solution for a European call option
## - Monte Carlo solution (Absorbing at zero)
## - Monte Carlo solution (Reflecting at zero)
## - Monte Carlo solution (Reflecting at zero + Milstein method)
## - Monte Carlo solution (Alfonsi)
## - Plot implied volality surface
##
## Dale Roberts <dale.roberts@anu.edu.au>
##
## PARAMETERS
##
## lambda: mean-reversion speed
## vbar: long-term average volatility
## eta: volatility of vol process
## rho: correlation between stock and vol
## v0: initial volatility
## r: risk-free interest rate
## tau: time to maturity
## S0: initial share price
## K: strike price
##
## MODEL
## 
## dS_t = S_t r dt + S_t sqrt(V_t)dW_t^S
## dV_t = \lambda (\vbar - V_t)dt - eta sqrt(V_t)dW_t^V
## with d<W^S,W^V>_t = \rho dt

ONEYEAR <- 250

Moneyness <- function(S, K, tau, r) {
    K*exp(-r*tau)/S
}

BlackScholesCall <- function(S0, K, tau, r, sigma, EPS=0.01) {
    d1 <- (log(S0/K) + (r + 0.5*sigma^2)*tau)/(sigma*sqrt(tau))
    d2 <- d1 - sigma*sqrt(tau)
    if (T < EPS) {
        return(max(S0-K,0))
    } else {
        return(S0*pnorm(d1) - K*exp(-r*(tau))*pnorm(d2))
    }
}

ImpliedVolCall <- function(S0, K, tau, r, price) {
    f <- function(x) BlackScholesCall(S0,K,tau,r,x) - price
    if (f(-1) * f(1) > 0)
        return(NA)
    uniroot(f,c(-1,1))$root
}

HestonCallClosedForm <-
    function(lambda, vbar, eta, rho, v0, r, tau, S0, K) {
	PIntegrand <- function(u, lambda, vbar, eta, rho, v0, r, tau, S0, K, j) {
            F <- S0*exp(r*tau)
            x <- log(F/K)
            a <- lambda * vbar
            
            if (j == 1) {
                b <- lambda - rho* eta
                alpha <- - u^2/2 - u/2 * 1i + 1i * u
                beta <- lambda - rho * eta - rho * eta * 1i * u
            } else {
                b <- lambda
                alpha <- - u^2/2 - u/2 * 1i
                beta <- lambda - rho * eta * 1i * u
            }
            
            gamma <- eta^2/2
            d <- sqrt(beta^2 - 4*alpha*gamma)
            rplus <- (beta + d)/(2*gamma)
            rminus <- (beta - d)/(2*gamma)
            g <- rminus / rplus
            
            D <- rminus * (1 - exp(-d*tau))/(1-g*exp(-d*tau))
            C <- lambda * (rminus * tau - 2/(eta^2) * log( (1-g*exp(-d*tau))/(1-g) ) )
            
            top <- exp(C*vbar + D*v0 + 1i*u*x)
            bottom <- (1i * u)
            Re(top/bottom)
	}
	
	P <- function(lambda, vbar, eta, rho, v0, r, tau, S0, K, j) {
            value <- integrate(PIntegrand, lower = 0, upper = Inf,
                               lambda, vbar, eta, rho, v0, r, tau,
                               S0, K, j, subdivisions=1000)$value
            0.5 + 1/pi * value
	}

        A <- S0*P(lambda, vbar, eta, rho, v0, r, tau, S0, K, 1)
        B <- K*exp(-r*tau)*P(lambda, vbar, eta, rho, v0, r, tau, S0, K, 0)
        A-B
    }

HestonCallMonteCarlo <-
    function(lambda, vbar, eta, rho, v0, r, tau, S0, K, nSteps=2000, nPaths=3000, vneg=2) {

        n <- nSteps
        N <- nPaths
        
        dt <- tau / n
        
        negCount <- 0
        
        S <- rep(S0,N)
        v <- rep(v0,N)
        
        for (i in 1:n)
            {
                W1 <- rnorm(N);
                W2 <- rnorm(N);
                W2 <- rho*W1 + sqrt(1 - rho^2)*W2;

                sqvdt <- sqrt(v*dt)
                S <- S*exp((r-v/2)*dt + sqrt(v * dt) * W1)
                
                if ((vneg == 3) & (2*lambda*vbar/(eta^2) <= 1)) {
                    cat("Variance not guaranteed to be positive with choice of lambda, vbar, and eta\n")
                    cat("Defaulting to Reflection + Milstein method\n")
                    vneg = 2
                }

                if (vneg == 0){
                    ## Absorbing condition
                    v <- v + lambda*(vbar - v)* dt + eta * sqvdt * W2
                    negCount <- negCount + length(v[v < 0])
                    v[v < 0] <- 0
                }
                if (vneg == 1){
                    ## Reflecting condition
                    sqvdt <- sqrt(v*dt)
                    v <- v + lambda*(vbar - v)* dt + eta * sqvdt * W2
                    negCount <- negCount + length(v[v < 0])
                    v <- ifelse(v<0, -v, v)
                }
                if (vneg == 2) {
                    ## Reflecting condition + Milstein
                    v <- (sqrt(v) + eta/2*sqrt(dt)*W2)^2 - lambda*(v-vbar)*dt - eta^2/4*dt
                    negCount <- negCount + length(v[v < 0])
                    v <- ifelse(v<0, -v, v)     
                }
                if (vneg == 3) {
                    ## Alfonsi - See Gatheral p.23
                    v <- v -lambda*(v-vbar)*dt +eta*sqrt(v*dt)*W2 - eta^2/2*dt      
                }
            }
        
        negCount <- negCount / (n*N);

        ## Evaluate mean call value for each path
        V <- exp(-r*tau)*(S>K)*(S - K); # Boundary condition for European call
        AV <- mean(V);
        AVdev <- 2 * sd(V) / sqrt(N);

        list(value=AV, lower = AV-AVdev, upper = AV+AVdev, zerohits = negCount)
    }

HestonSurface <- function(lambda, vbar, eta, rho, v0, r, tau, S0, K, N=5, min.tau = 1/ONEYEAR) {
    LogStrikes <- seq(-0.5, 0.5, length=N)
    Ks <- rep(0.0,N)
    taus <- seq(min.tau, tau, length=N)
    vols <- matrix(0,N,N)

    TTM <- Money <- Vol <- rep(0,N*N)
    
    HestonPrice <- function(K, tau) {
        HestonCallClosedForm(lambda, vbar, eta, rho, v0, r, tau, S0, K)
    }

    n <- 1
    for (i in 1:N) {
        for (j in 1:N) {
            Ks[i] <- exp(r * taus[j]+LogStrikes[i]) * S0
            price <- HestonPrice(Ks[i],taus[j])
            iv <- ImpliedVolCall(S0, Ks[i], taus[j], r, price)
            TTM[n] <- taus[j] * ONEYEAR # in days
            Money[n] <- Moneyness(S0,Ks[i],taus[j],r)
            Vol[n] <- iv
            n <- n+1
        }
    }

    data.frame(TTM=TTM, Moneyness=Money, ImpliedVol=Vol)
}

PlotHestonSurface <-
    function(lambda=6.21, vbar=0.019, eta=0.61, rho=-0.7, v0=0.010201, r=0.0319,
             tau=1.0, S0=100, K=100, N=30, min.tau = 1/ONEYEAR, ...) {
        
        Ks <- seq(0.8*K, 1.25 * K, length=N)  
        taus <- seq(0.21, tau, length=N)
        
        HestonPrice <- Vectorize(function(k, t) {
            HestonCallClosedForm(lambda, vbar, eta, rho, v0, r, t, S0, k)})
        
        IVHeston <- Vectorize(function(k,t) { ImpliedVolCall(S0, k, t, r, HestonPrice(k,t))})
        
        z <- outer(Ks, taus, IVHeston)
        
        nrz <- nrow(z)
        ncz <- ncol(z)
        nb.col <- 256
        color <- heat.colors(nb.col)
        facet <- - (z[-1, -1] + z[-1, -ncz] + z[-nrz, -1] + z[-nrz, -ncz])
        facetcol <- cut(facet, nb.col)
        
        persp(x=Ks, y=taus, z, theta = 40, phi = 20, expand = 0.5, col=color[facetcol],
              xlab="Strikes", ylab="Time to maturity", zlab="Implied Volatility",
              ticktype="detailed", ...) -> res

        return(invisible(z))
    }
