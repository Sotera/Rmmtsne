#' Multiple maps t-SNE with symmetric probability matrix
#'
#' \code{mmtsneP} estimates a multiple maps t-distributed stochastic neighbor
#'    embedding (multiple maps t-SNE) model.
#'
#' This code is an almost direct port of the original multiple maps t-SNE Matlab
#'    code by van der Maaten and Hinton (2012). \code{mmtsne} estimates a
#'    multidimensional array of \code{N x no_dims x no_maps}. Each map is an
#'    \code{N x no_dims} matrix of estimated t-SNE coordinates. When
#'    \code{no_maps=1}, multiple maps t-SNE reduces to standard t-SNE.
#'
#' @param P An \eqn{N x N} symmetric joint probability distribution matrix.
#'    These can be constructed from an \eqn{N} by \eqn{D} matrix with
#'    \code{\link{x2p}} and \code{\link{p2sp}}. Alternatively, the wrapper
#'    function \code{\link{mmtsne}} will wrap the matrix construction and
#'    multiple maps t-SNE model estimation into a single step.
#' @param no_maps The number of maps (positive whole number) to be estimated.
#' @param no_dims The number of dimensions per map. Typical values are 2 or 3.
#' @param max_iter The number of iterations to run.
#' @param momentum Constant scaling factor for update momentum in gradient
#'    descent algorithm.
#' @param final_momentum Constant scaling factor for update momentum in gradient
#'    descent algorithm after the momentum switch point.
#' @param eps A small positive value near zero.
#' @param mom_switch_iter The iteration at which momentum switches from
#'    \code{momentum} to \code{final_momentum}.
#'
#' @return A list that includes the following objects:
#'    \describe{
#'    \item{Y}{An \code{N x no_dims x no_maps} array of predicted coordinates.}
#'    \item{weights}{An \code{N x no_maps} matrix of unscaled weights. A high
#'        weight on entry \eqn{i, j} indicates a greater contribution of point
#'        \eqn{i} on map \eqn{j}.}
#'    \item{proportions}{An \code{N x no_maps} matrix of scaled weights. A high
#'        weight on entry \eqn{i, j} indicates a greater contribution of point
#'        \eqn{i} on map \eqn{j}.}
#'    }
#'
#' @examples
#' # Load the iris dataset
#' data("iris")
#'
#' # Produce a symmetric joint probability matrix
#' prob_matrix <- p2sp(x2p(as.matrix(iris[,1:4])))
#'
#' # Estimate a mmtsne model with 2 maps, 2 dimensions each
#' model <- mmtsneP(prob_matrix, no_maps=2)
#'
#' # Plot the results side-by-side for inspection
#' # Points scaled by map proportion weights plus constant factor
#' par(mfrow=c(1,2))
#' plot(model$Y[,,1], col=iris$Species, cex=model$proportions[,1] + 0.2)
#' plot(model$Y[,,2], col=iris$Species, cex=model$proportions[,2] + 0.2)
#' par(mfrow=c(1,1))
#'
#' @references L.J.P. van der Maaten and G.E. Hinton. ``Visualizing Non-Metric
#'    Similarities in Multiple Maps.'' \emph{Machine Learning} 87(1):33-55,
#'    2012.
#'    \href{https://lvdmaaten.github.io/publications/papers/
#'    MachLearn_2012.pdf}{PDF.}
#'
#' @export
mmtsneP <- function(P, no_maps, no_dims=2, max_iter=500, momentum=0.5, final_momentum=0.8, mom_switch_iter=250, eps=1e-7)
{
  n <- nrow(P)
  epsilonY <- 250
  epsilonW <- 100

  P <- P * 4

  ## Initialize random solution
  Y <- array(rnorm(n * no_dims * no_maps, 0, 1) * 0.001, c(n, no_dims, no_maps))
  y_incs <- array(0, c(n, no_dims, no_maps))
  weights <- matrix(1/no_maps, nrow=n, ncol=no_maps)

  ## Pre-allocate objects
  num <- array(0, c(n, n, no_maps))
  QQ <- array(0, c(n, n, no_maps))
  dCdP <- array(0, c(n, no_maps))
  dCdD <- array(0, c(n, n, no_maps))
  dCdY <- array(0, c(n, no_dims, no_maps))

  ## Run the iterations
  for(iter in 1:max_iter)
  {
    ## Compute the mixture proportions from the mixture weights
    proportions <- exp(-weights) ## n x no_maps
    proportions <- proportions / matrix(rowSums(proportions),byrow=F,nrow=n,ncol=no_maps) ## n x no_maps

    ## Compute pairwise affinities per map
    for(m in 1:no_maps)
    {
      sum_Y <- rowSums(Y[,,m] ^ 2) ## n x 1
      sum_Y <- matrix(sum_Y, byrow=F, nrow=n, ncol=n) ## n x n

      tmp <- 1 / (1 + sum_Y + (t(sum_Y) + (-2 * Y[,,m] %*% t(Y[,,m])) ) ) ## n x n
      diag(tmp) <- 0
      num[,,m] <- tmp ## n x n x no_maps
    }

    ## Compute pairwise affinities under the mixture model
    QZ <- matrix(eps, nrow=n, ncol=n) ## n x n
    for(m in 1:no_maps)
    {
      QQ[,,m] <- (sum(proportions[,m] * proportions[,m])) * num[,,m] ## n x n
      QZ <- QZ + QQ[,,m] ## n x n
    }
    Z <- sum(QZ) ## 1 x 1
    Q <- QZ / Z ## n x n

    ## Compute the derivative of cost function w.r.t. mixture proportions
    PQ <- Q - P ## n x n
    tmp <- (1 / QZ) * PQ ## n x n
    for(m in 1:no_maps)
    {
      ## CHECK THIS LINE FOR byrow DIRECTIONALITY
      dCdP[,m] <- colSums(matrix(proportions[,m], byrow=F, nrow=n, ncol=n) * (num[,,m] * tmp))
    }
    dCdP <- 2 * dCdP

    ## Compute derivative of cost function w.r.t. mixture weights
    dCdW <- proportions * (matrix(rowSums(dCdP * proportions), byrow=F, nrow=n, ncol=no_maps) - dCdP)

    ## Computer derivative of cost function w.r.t. pairwise distances
    for(m in 1:no_maps)
    {
      dCdD[,,m] <- (QQ[,,m] / QZ) * -PQ * num[,,m]
    }

    ## Compute derivative of cost function w.r.t. the maps
    for(m in 1:no_maps)
    {
      for(i in 1:n)
      {
        dCdY[i,,m] <- colSums(matrix((dCdD[i,,m] + dCdD[,i,m]), byrow=F, nrow=n, ncol=no_dims) * (matrix(Y[i,,m],byrow=T,nrow=n,ncol=no_dims) - Y[,,m]))
      }
    }

    ## Update the solution
    y_incs <- momentum * y_incs - epsilonY * dCdY

    Y <- Y + y_incs
    for(m in 1:no_maps)
    {
      Y[,,m] <- Y[,,m] - matrix(colMeans(Y[,,m]), byrow=T, nrow=n, ncol=no_dims)
    }

    weights <- weights - (epsilonW * dCdW)

    ## Update momentum if necessary
    if(iter == mom_switch_iter)
      momentum = final_momentum
    if(iter == 50)
      P = P / 4

    ## Compute the value of the cost function
    if(iter %% 25 == 0)
    {
      tmp_P <- P
      tmp_Q <- Q
      tmp_P[tmp_P <= 0] <- 10e-10
      tmp_Q[tmp_Q <= 0] <- 10e-10
      C <- sum(tmp_P * log(tmp_P / tmp_Q))
      cat("\rIteration ",iter,": error is ",C)
    }
  }
  return(list("Y"=Y,"weights"=weights,"proportions"=proportions))
}

#' Multiple maps t-SNE
#'
#' \code{mmtsne} estimates a multiple maps t-distributed stochastic neighbor
#'    embedding (multiple maps t-SNE) model.
#'
#' \code{mmtsne} is a wrapper that performs multiple maps t-SNE on an input
#'    dataset, \code{X}. The function will pre-process \code{X}, an \eqn{N}
#'    by \eqn{D} matrix or dataframe, then call \code{\link{mmtsneP}}.
#'    The pre-processing steps include calls to \code{\link{x2p}} and
#'    \code{\link{p2sp}} to convert \code{X} into an \eqn{N} by \eqn{N}
#'    symmetrical joint probability matrix.
#'
#' The \code{mmtnsep} code is an almost direct port of the original multiple
#'    maps t-SNE Matlab code by van der Maaten and Hinton (2012). \code{mmtsne}
#'    estimates a multidimensional array of \code{N x no_dims x no_maps}. Each
#'    map is an \code{N x no_dims} matrix of estimated t-SNE coordinates. When
#'    \code{no_maps=1}, multiple maps t-SNE reduces to standard t-SNE.
#'
#' @param X A dataframe or matrix of \eqn{N} rows and \eqn{D} columns.
#' @param no_maps The number of maps (positive whole number) to be estimated.
#' @param no_dims The number of dimensions per map. Typical values are 2 or 3.
#' @param perplexity The target perplexity for probability matrix
#'    construction. Commonly recommended values range from 5 to 30.
#'    Perplexity roughly corresponds to the expected number of neighbors
#'    per data point.
#' @param max_iter The number of iterations to run.
#' @param momentum Constant scaling factor for update momentum in gradient
#'    descent algorithm.
#' @param final_momentum Constant scaling factor for update momentum in gradient
#'    descent algorithm after the momentum switch point.
#' @param eps A small positive value near zero.
#' @param mom_switch_iter The iteration at which momentum switches from
#'    \code{momentum} to \code{final_momentum}.
#'
#' @return A list that includes the following objects:
#'    \describe{
#'    \item{Y}{An \code{N x no_dims x no_maps} array of predicted coordinates.}
#'    \item{weights}{An \code{N x no_maps} matrix of unscaled weights. A high
#'        weight on entry \eqn{i, j} indicates a greater contribution of point
#'        \eqn{i} on map \eqn{j}.}
#'    \item{proportions}{An \code{N x no_maps} matrix of scaled weights. A high
#'        weight on entry \eqn{i, j} indicates a greater contribution of point
#'        \eqn{i} on map \eqn{j}.}
#'    }
#'
#' @examples
#' # Load the iris dataset
#' data("iris")
#'
#' # Estimate a mmtsne model with 2 maps, 2 dimensions each
#' model <- mmtsne(iris[,1:4], no_maps=2)
#'
#' # Plot the results side-by-side for inspection
#' # Points scaled by map proportion weights plus constant factor
#' par(mfrow=c(1,2))
#' plot(model$Y[,,1], col=iris$Species, cex=model$proportions[,1] + .2)
#' plot(model$Y[,,2], col=iris$Species, cex=model$proportions[,2] + .2)
#' par(mfrow=c(1,1))
#'
#' @references L.J.P. van der Maaten and G.E. Hinton. ``Visualizing Non-Metric
#'    Similarities in Multiple Maps.'' \emph{Machine Learning} 87(1):33-55,
#'    2012.
#'    \href{https://lvdmaaten.github.io/publications/papers/
#'    MachLearn_2012.pdf}{PDF.}
#'
#' @export
mmtsne <- function(X, no_maps=1, no_dims=2, perplexity=30, max_iter=500, momentum=0.5, final_momentum=0.8, mom_switch_iter=250, eps=1e-7)
{
  sym_matrix <- p2sp(x2p(as.matrix(X, perplexity=perplexity)))
  mmtsneP(sym_matrix, no_maps, no_dims, max_iter, momentum, final_momentum, mom_switch_iter, eps)
}
