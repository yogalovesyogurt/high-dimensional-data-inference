library(msaenet)
dat <- msaenet.sim.gaussian(
  n = 150, p = 500, rho = 0.75,
  coef = c(1.2,1.2,3.5,3.5,1.2,1.2,1.2,3.5,3.5,1.2,1.2,1.2,3.5,3.5,1.2,1.2,1.2,3.5,3.5,1.2,1.2,1.2,3.5,3.5,1.2), snr = 10, p.train = 2/3,
  seed = 100
)
msaenet.fit <- msaenet(
  dat$x.tr, dat$y.tr,
  alphas = seq(0.1, 0.9, 0.1),
  nsteps = 5L, tune.nsteps = "ebic",
  seed = 1001
)
msaenet.pred <- predict(msaenet.fit, dat$x.te)
msaenet.rmse(dat$y.te, msaenet.pred)
msaenet.nzv(msaenet.fit)
msaenet.fp(msaenet.fit,1:500)
#coef(msaenet.fit)
aenet.fit <- aenet(
  dat$x.tr, dat$y.tr,
  alphas = seq(0.1, 0.9, 0.1), seed=1002
)
aenet.pred <- predict(aenet.fit, dat$x.te)
msaenet.rmse(dat$y.te, aenet.pred)
msaenet.nzv(aenet.fit)
msaenet.fp(aenet.fit,1:500)

alasso.fit <- aenet(
  dat$x.tr, dat$y.tr,
  alphas = c(1), seed=10
)
alasso.pred <- predict(alasso.fit, dat$x.te)
msaenet.rmse(dat$y.te, alasso.pred)
msaenet.nzv(alasso.fit)
msaenet.fp(alasso.fit,1:500)

library(glmnet)
lassolam=cv.glmnet(dat$x.tr, dat$y.tr,nfolds = 5)$lambda.min
lasso.fit=glmnet(dat$x.tr, dat$y.tr,lambda = lassolam)
lasso.pred=predict(lasso.fit,dat$x.te)
msaenet.rmse(dat$y.te, lasso.pred)
lassocoef=coef(lasso.fit)

enetlam=cv.glmnet(dat$x.tr, dat$y.tr,nfolds = 5,alphas = seq(0.1, 0.9, 0.1))$lambda.min
enet.fit=glmnet(dat$x.tr, dat$y.tr,lambda = enetlam,alphas = seq(0.1, 0.9, 0.1))
enet.pred=predict(enet.fit,dat$x.te)
msaenet.rmse(dat$y.te, enet.pred)
enetcoef=coef.glmnet(enet.fit)

