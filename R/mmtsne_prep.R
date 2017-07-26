#' Estimate perplexity and probability values
#'
#' \code{hbeta} returns the perplexity and probability values for a row
#'    of data \code{D}.
hbeta <- function(D, beta=1)
{
  ## Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution.
  P <- exp(-D * beta)
  sumP <- sum(P)
  H <- log(sumP) + beta * sum(D * P) / sumP
  P <- P / sumP
  list("H"=H,"P"=P)
}

#' Data to probability matrix
#'
#' \code{x2p} returns a pair-wise conditional probability matrix given an input
#'    matrix \emph{X}.
#'
#' This function is an almost direct port of the original Python implementation
#' by van der Maaten and Hinton (2008). It uses a binary search to estimate
#' probability values for all pairwise-elements of \code{X}. The conditional
#' Gaussian distributions should all be of equal perplexity.
#'
#' @param X A data matrix with \eqn{N} rows.
#' @param perplexity The target perplexity. Values between 5 and 50 are
#'    generally considered appropriate. Loosely translates into the
#'    expected number of neighbors per point.
#' @param tol A small positive value.
#' @return An \code{N x N} matrix of pair-wise probabilities.
#'
#' @references L.J.P. van der Maaten and G.E. Hinton. ``Visualizing
#'    High-Dimensional Data Using t-SNE.'' \emph{Journal of Machine Learning
#'    Research} 9(Nov):2579-2605, 2008. \href{https://lvdmaaten.github.io/
#'    publications/papers/JMLR_2008.pdf}{PDF.}
x2p <- function(X, perplexity=30, tol=1e-5)
{
  n <- nrow(X)
  d <- ncol(X)
  sum_X <- rowSums(X^2)
  D <- t(-2 * X %*% t(X) + matrix(sum_X, byrow=F, nrow=n, ncol=n)) + matrix(sum_X, byrow=F, nrow=n, ncol=n)
  P <- matrix(0, nrow=n, ncol=n)
  beta <- matrix(1, nrow=n, ncol=1)
  logU <- log(perplexity)

  for(i in 1:n)
  {
    if(i %% 500 == 0)
      cat("\rComputing P-values for point",i,"of",n,"...")

    notI <- (1:n)[!(1:n) %in% i]
    betamin <- -Inf
    betamax <- Inf
    Di <- D[i, notI]
    hbeta <- hbeta(Di, beta[i])
    H <- hbeta$H
    thisP <- hbeta$P

    Hdiff <- H - logU
    tries <- 0
    while(abs(Hdiff) > tol & tries < 50)
    {
      if(Hdiff > 0)
      {
        betamin <- beta[i]
        if(betamax == Inf | betamax == -Inf)
        {
          beta[i] <- beta[i] * 2
        }else{
          beta[i] <- (beta[i] + betamax) / 2
        }
      }else{
        betamax <- beta[i]
        if(betamin == Inf | betamin == -Inf)
        {
          beta[i] <- beta[i] /2
        }else{
          beta[i] <- (beta[i] + betamin) / 2
        }
      }
      hbeta <- hbeta(Di, beta[i])
      H <- hbeta$H
      thisP <- hbeta$P
      Hdiff <- H - logU
      tries <- tries + 1
    }
    P[i, notI] <- thisP
  }
  cat("\rMean value of sigma:",mean(sqrt(1/beta)))
  P
}

#' Probability matrix to symmetric probability matrix
#'
#' \code{p2sp} returns a symmetrical pair-wise joint probability
#'    matrix given an input probability matrix \emph{P}.
#'
#' @param P An \code{N x N} probability matrix, like those produced by
#'  \code{\link{x2p}}
#' @return An \code{N x N} symmetrical matrix of pair-wise probabilities.
p2sp <- function(P)
{
  P <- P + t(P)
  P <- P / sum(P)
  P[P <= 0] <- 1e-10
  P
}




