import numpy as np
from scipy.special import gammaln, digamma
from numpy.linalg import slogdet
import sys
import random
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
#np.set_printoptions(threshold=np.nan)


def c_lmvgamma (x, p):
  ans =(p * (p - 1)/4.0) * np.log(np.pi)
  for i in range(p):
    ans += gammaln(x  - i/2.0)
  return ans

def c_mvdigamma (x, p):
  ans = 0.
  for i in range(p):
    ans += digamma(x  - i/2.0) 
  return ans


def logsumexp(x):
  s, offset = 0, np.max(x)
  for i in range(x.shape[0]):
    s+=np.exp(x[i]-offset)
  return np.log(s) + offset

# def C_VI_PYMMG(x, max_it, Kmax, tol,  theta0, tau0,
#       b0, Omega0, alpha, d, m, D, phi, st):
#   #Variational Inference of Pitman-Yor Mixture of Multivariate Gaussian
#   n, p =x.shape[:2]
#   #variational parameters
#   Psi, c = np.zeros((p,p,Kmax)), np.zeros(Kmax)
#   c.fill(p)
#   gamma1 = np.ones(Kmax)
#   gamma2 = np.zeros(Kmax)
#   gamma2.fill(alpha)
#   nk=np.sum(phi,axis=0)
#   o, vtmp = np.zeros(Kmax), np.zeros(p)
#   if (st==1):
#     o=np.argsort(nk[::-1]) 
#   else:
#     o=np.linspace(0,Kmax-1,Kmax)
  
#   it, flag=0,1
#   #tmp values for elbo
#   e1, e2, e3 = np.zeros(Kmax), np.zeros(Kmax), np.zeros(Kmax)
#   print("VB starts")
#   elbo = np.zeros(max_it)
  
#   for kk in range(Kmax):
#     k=o[kk]
#     Psi[:,:,k]=D[:,:,k]/(tau0+nk[k])/c[k]
#     m[:,k]= tau0*theta0 + np.sum(x*phi[:,k][:,np.newaxis],0).T/(tau0+nk[k]) 
#     c[k]=b0+nk[k]
#     vtmp=m[:,k]-theta0
#     D[:,:,k]=Omega0+tau0@vtmp@vtmp.T+(tau0+nk[k])@Psi[:,:,k]
#     for i in range(n):
#       vtmp=x[i,:].T-m[:,k]
#       D[:,:,k] += phi[i,k]@vtmp@vtmp.T

#   for kk in range(Kmax-1):
#     k=o[kk]
#     gamma1[k]=1.0-d+nk[k]
#     gamma2[k]=alpha+d*(kk+1)+np.sum(nk[o[(kk+1):Kmax]]) 
  
  
#   for kk in range(Kmax-1):
#     k=o[kk]
#     e3[k]=digamma(gamma2[k])-digamma(gamma1[k]+gamma2[k])
  
#   for kk in range(Kmax-1):
#     k=o[kk]
#     sign, det_val = slogdet(D[:,:,k])
#     e1[k]=digamma(gamma1[k])-digamma(gamma1[k]+gamma2[k])
#     e2[k]=.5*(c_mvdigamma(c[k]/2,p)-det_val)
#     dtmp=e1[k]+e2[k]
#     if (kk>0):
#       dtmp += np.sum(e3[o[0:kk]])
#     phi[:,k].fill(dtmp)
 
#   kk, k=Kmax-1, o[kk]
#   sign, det_val = slogdet(D[:,:,k]) 
#   dtmp=.5*(c_mvdigamma(c[k]/2,p)-det_val)
#   for kkk in range(kk):
#     dtmp+=digamma(gamma2[o[kkk]])-digamma(gamma1[o[kkk]]+gamma2[o[kkk]])
#   phi[:,k].fill(dtmp)
  
#   for i in range(n):
#     for kk in range(Kmax):
#       k=o[kk]
#       vtmp=x[i,:].T-m[:,k]
#       phi[i,k]+=-.5*c[k]*np.trace(np.linalg.solve(D[:,:k],vtmp@vtmp.T+Psi[:,:,k]))
    
#     phi[i,:]=np.exp(phi[i,:]-logsumexp(phi[i,:]))
  
#   nk=np.sum(phi,0)
#   if (st==1): #if sort
#     o=np.argsort(nk[::-1])
  
  
#   while (it<max_it and flag==1):
#     for kk in range(Kmax-1):
#       k=o[kk]
#       e3[k]=digamma(gamma2[k])- digamma(gamma1[k]+gamma2[k])
    
#     for kk in range(Kmax-1):
#       k=o[kk]
#       sign, det_val = slogdet(D[:,:,k])
#       e1[k]= digamma(gamma1[k])- digamma(gamma1[k]+gamma2[k])
#       e2[k]=.5*(c_mvdigamma(c[k]/2,p)-det_val)
#       dtmp=e1[k]+e2[k]
#       if (kk>0):
#         dtmp+=np.sum(e3[o[:kk]]) 
#       phi[:,k].fill(dtmp) 

#     kk, k=Kmax-1, o[kk]
#     sign, det_val = slogdet(D[:,:,k])
#     e2[k]=.5*(c_mvdigamma(c[k]/2,p)-det_val)
#     dtmp=e2[k]
#     if (kk>0):
#       dtmp+=np.sum(e3[o[:kk]]) 
   
#     phi[:,k].fill(dtmp)
    
#     for i in range(n):
#       for kk in range(Kmax):
#         k=o[kk]
#         vtmp=x[i,:].T-m[:,k]
#         phi[i,k]+=-.5*c[k]*np.trace(np.linalg.solve(D[:,:k],vtmp@vtmp.T+Psi[:,:,k]))
#       phi[i,:]=np.exp(phi[i,:]-logsumexp(np.exp(phi[i,:])))
  
#     nk=np.sum(phi,0)
#     if (st==1):#if sort
#       o=np.argsort(nk[::-1])

#     for kk in range(Kmax):
#       k=o[kk]
#       Psi[:,:,k]=D[:,:,k]/(tau0+nk[k])/c[k]
#       m[:,k]=(tau0*theta0+np.sum(x*phi[:,k][:,np.newaxis],0).T)/(tau0+nk[k])
#       c[k]=b0+nk[k]
#       vtmp=m[:,k]-theta0
#       D[:,:,k]=Omega0+tau0@vtmp@vtmp.T+(tau0+nk[k])*Psi[:,:,k]
#       for i in range(n):
#         vtmp=x[i:,].T-m[:,k]
#         D[:,:,k] += phi[i,k]@vtmp@vtmp.T

#     for kk in range(Kmax-1):
#       k=o[kk]
#       gamma1[k]=1.0-d+nk[k]
#       gamma2[k]=alpha+d*[kk+1]+np.sum(nk[o[(kk+1):Kmax]])
    
#     for kk in range(Kmax-1):
#       k=o[kk]
#       elbo[it] += (alpha+d*(kk+1)-gamma2[k]+np.sum(nk[o[kk+1,Kmax]]))*e3[k]+(nk[k]+1-gamma1[k])*e1[k]+gammaln(gamma1[k])+gammaln(gamma2[k])- gammaln(gamma1[k]+gamma2[k])
    
#     for kk in range(Kmax):
#       k=o[kk]
#       sign, det_val = slogdet(Psi[:,:,k])   
#       elbo[it] += (b0+nk[k]-c[k] +1)*e2[k]+.5*det_val-np.sum(phi[:,k]*np.log(phi[:,k]+10**(-10)))+np.log(2.0)/2.0*c[k]*p+c_lmvgamma(c[k]/2,p)
#       sign, det_val = slogdet(D[:,:,k])   
#       elbo[it] -= c[k]/2*det_val
    
#     if it>0 and abs(elbo[it]-elbo[it-1])<tol:
#       flag=0
#     it+=1
  
#   if (flag==1):
#     print("warning: not convengent")
  
#   xi_est=np.argmax(phi,1) 
#   labels=np.unique(xi_est) 
#   return xi_est, phi, m, D, labels, elbo[:it]

def C_VI_PYMMG_CoC(xx,  SS, n_vec, max_it,  tol,  Kmax,  theta0, tau0,
           b0, Omega0,  alpha,  d,  m,  D, phi, st):
   #Variational Inference of Pitman-Yor Mixture of Multivariate Gaussian with Clustering of Clusters
  n, p=xx.shape[:2]  
  #variational parameters
  Psi = np.zeros((p,p,Kmax))
  c = np.zeros(Kmax)
  c.fill(p)
  gamma1 = np.ones(Kmax)
  gamma2 = np.zeros(Kmax)
  gamma2.fill(alpha)
  gamma1_old = np.ones(Kmax)
  gamma2_old = np.zeros(Kmax)
  gamma2_old.fill(alpha)
  nk = np.sum(phi*n_vec[:,np.newaxis],0) ##
  o, o_old = np.zeros(Kmax), np.zeros(Kmax)
  vtmp = np.zeros(p)
  if (st==1):
    o=np.argsort(nk[::-1]) 
  else:
    o=np.linspace(0,Kmax-1,Kmax)

  it = 0
  j = np.zeros(1)
  mtmp = np.zeros((p,p))
  e1, e2, e3 = np.zeros(Kmax), np.zeros(Kmax), np.zeros(Kmax) #tmp values for elbo
  print("SIGN starts")
  flag=1
  elbo = np.zeros(max_it)
  for kk in range(Kmax):
    k=o[kk]
    Psi[:,:,k]=D[:,:,k]/(tau0+nk[k])/c[k]
    m[:,k]=(tau0*theta0+np.sum(xx*phi[:,k][:,np.newaxis],0).T)/(tau0+nk[k]) ##
    c[k]=b0+nk[k]
    vtmp=m[:,k]-theta0
    D[:,:,k]=Omega0+tau0*vtmp[:,np.newaxis]@vtmp[:,np.newaxis].T+(tau0+nk[k])*Psi[:,:,k]
    for i in range(n):
      mtmp=m[:,k][:,np.newaxis]@xx[i,:][np.newaxis,:]
      D[:,:,k] += phi[i,k]*(SS[:,:,i]-mtmp-mtmp.T+n_vec[i]*m[:,k][:,np.newaxis]@m[:,k][:,np.newaxis].T)

  for kk in range(Kmax-1):
    k=o[kk]
    gamma1[k]=1.0-d+nk[k]
    gamma2[k]=alpha+d*(kk+1)+np.sum(nk[o[(kk+1):Kmax]])
  
  for kk in range(Kmax-1):
    k=o[kk]
    sign, det_val = slogdet(D[:,:,k])   
    phi[:,k]=n_vec*(digamma(gamma1[k])-digamma(gamma1[k]+gamma2[k])+.5*(c_mvdigamma(c[k]/2,p)-det_val))
    for kkk in range(kk):
      phi[:,k] += n_vec*(digamma(gamma2[o[kkk]])-digamma(gamma1[o[kkk]]+gamma2[o[kkk]]))

  kk=Kmax-1
  k=o[kk]
  sign, det_val = slogdet(D[:,:,k])   
  phi[:,k]=n_vec*(.5*(c_mvdigamma(c[k]/2,p)-det_val))
  for kkk in range(kk):
    phi[:,k]+=n_vec*(digamma(gamma2[o[kkk]])- digamma(gamma1[o[kkk]]+gamma2[o[kkk]]))
  
  for i in range(n):
    for kk in range(Kmax):
      k=o[kk]
      mtmp=m[:,k][:,np.newaxis]@xx[i,:][np.newaxis,:]
      phi[i,k] += -.5*c[k]*np.trace(np.linalg.solve(
      	D[:,:,k],SS[:,:,i]-mtmp-mtmp.T+n_vec[i]*m[:,k][:,np.newaxis]@m[:,k][:,np.newaxis].T+n_vec[i]*Psi[:,:,k]))
   
    phi[i,:]=np.exp(phi[i,:]-logsumexp(phi[i,:]))
 
  nk=np.sum(phi*n_vec[:,np.newaxis],0)
  if (st==1): #if sort
    o=np.argsort(nk[::-1])
  
  
  while (it<max_it and flag==1):
    phi_old=phi
    for kk in range(Kmax-1):
      k=o[kk]
      e3[k]=digamma(gamma2[k])- digamma(gamma1[k]+gamma2[k])
    
    for kk in range(Kmax-1):
      k=o[kk]
      sign, det_val = slogdet(D[:,:,k])  
      e1[k]= digamma(gamma1[k])- digamma(gamma1[k]+gamma2[k])
      e2[k]=.5*(c_mvdigamma(c[k]/2,p)-det_val)
      phi[:,k]=n_vec*(e1[k]+e2[k])
      if (kk>0):
        phi[:,k]+=n_vec*np.sum(e3[o[:kk]])

    kk = Kmax-1
    k= o[kk]
    sign, det_val = slogdet(D[:,:,k])   
    e2[k]=.5*(c_mvdigamma(c[k]/2,p)-det_val)
    phi[:,k]=n_vec*e2[k]
    if (kk>0):
      phi[:,k]+=n_vec*np.sum(e3[o[:kk]])
    
    for i in range(n):
      for kk in range(Kmax):
        k=o[kk]
        mtmp=m[:,k][:,np.newaxis]@xx[i,:][np.newaxis,:]
        phi[i,k]+=-.5*c[k]*np.trace(np.linalg.solve(
        	D[:,:,k],SS[:,:,i]-mtmp-mtmp.T+n_vec[i]*m[:,k][:,np.newaxis]@m[:,k][:,np.newaxis].T+n_vec[i]*Psi[:,:,k]))
      phi[i,:]=np.exp(phi[i,:]-logsumexp(phi[i,:]))
    
    nk=np.sum(phi*n_vec[:,np.newaxis],0)
    o_old=np.copy(o)
    if (st==1): #if sort
      o=np.argsort(nk[::-1])
    
    
    for kk in range(Kmax):
      k=o[kk]
      Psi[:,:,k]=D[:,:,k]/(tau0+nk[k])/c[k]
      m[:,k] = (tau0*theta0+np.sum(xx*phi[:,k][:,np.newaxis],0))/(tau0+nk[k])
      c[k]=b0+nk[k]
      vtmp=m[:,k]-theta0
      D[:,:,k]=Omega0+tau0*vtmp[:,np.newaxis]@vtmp[:,np.newaxis].T+(tau0+nk[k])*Psi[:,:,k]
      for i in range(n):
        mtmp=m[:,k][:,np.newaxis]@xx[i,:][np.newaxis,:]
        D[:,:,k] += phi[i,k]*(SS[:,:,i]-mtmp-mtmp.T+n_vec[i]*m[:,k][:,np.newaxis]@m[:,k][:,np.newaxis].T)
    gamma1_old=gamma1
    gamma2_old=gamma2
    for kk in range(Kmax-1):
      k=o[kk]
      gamma1[k]=1.0-d+nk[k]
      gamma2[k]=alpha+d*(kk+1)+np.sum(nk[o[kk+1:Kmax]])
    para_diff=np.sum(np.abs(gamma1_old[o_old[0:(Kmax-1)]]-gamma1[o[0:(Kmax-1)]])+np.abs(gamma2_old[o_old[0:(Kmax-1)]]-gamma2[o[0:(Kmax-1)]]))+np.sum(np.abs(phi_old-phi))

    for kk in range(Kmax-1):
      k=o[kk]
      elbo[it] += (alpha+d*(kk+1)-gamma2[k]+np.sum(nk[o[(kk+1):Kmax]]))*e3[k]+(nk[k]+1-gamma1[k])*e1[k]+ \
      gammaln(gamma1[k])+ gammaln(gamma2[k])- gammaln(gamma1[k]+gamma2[k])
   
    for kk in range(Kmax):
      k = o[kk]
      sign, det_val = slogdet(Psi[:,:,k])
      elbo[it] += (b0+nk[k]-c[k]+1)*e2[k]+.5*det_val-\
      		np.sum(phi[:,k]*np.log(phi[:,k]+10**(-10)))+np.log(2.0)/2.0*c[k]*p+c_lmvgamma(c[k]/2,p)
      sign, det_val = slogdet(D[:,:,k])   
      elbo[it] -= c[k]/2*det_val
    
 
    if (it>0 and abs(elbo[it]-elbo[it-1])<tol and para_diff<tol):
      flag=0
    it += 1
 
  if (flag==1):
    print("warning: not convengent")

  xi_est=np.argmax(phi,1)
  labels=np.unique(xi_est)

  ## get the estimated precision matrix
  Prec = np.zeros((p,p,Kmax))
  for kk in range(Kmax):
    Prec[:,:,kk] = np.linalg.inv(D[:,:,kk])*c[kk]
  return xi_est, phi, m, D, labels, elbo[:it], para_diff, Prec




def R_VI_PYMMG_CoC(SS, xx, n_vec,Kmax=None,max_it=200,tol=10**(-2),
                  theta0=None,tau0=1,b0=None,Omega0=None,
                  alpha=1,d=0.5,ini=True,st=True,seed=1):

  if theta0 == None:
    theta0 = np.zeros(xx.shape[1])
  if b0 == None:
    b0 = xx.shape[1]
  if Omega0 == None:
    Omega0=np.identity(xx.shape[1])/b0



    
  #Variational Inference of Pitman-Yor Mixture of Multivariate Gaussian with 
  #Clustering of Clusters
  random.seed(seed)
  n, p = xx.shape[:2] 

  if Kmax == None:
    if n>1000:
      Kmax=n/50
    elif n<20:
      Kmax=n
    else:
      Kmax=20
   
  #cluster clusters
  xm=xx/n_vec[:,np.newaxis]
  #Initialize variational parameters
  if ini:
    if n>Kmax:
      k_KM=Kmax
      KM=KMeans(n_clusters=k_KM, random_state = 25).fit(xm)
      #print(KM.labels_)
      #print(KM.labels_.shape)
      phi_KM=np.zeros((n,k_KM))
      for i in range(n):
        phi_KM[i,KM.labels_[i]]=1
      m = np.zeros((p,Kmax))
      D = np.stack([np.identity(p)*p for ip in range(Kmax)], axis=2)
      for k in range(k_KM):
        ind=(phi_KM[:,k]==1)
        if np.sum(ind)>1:
          m[:,k]=np.sum(xx[ind,:], axis=0)/np.sum(n_vec[ind])
          D[:,:,k]=((np.sum(SS[:,:,ind],2)-m[:,k][:, np.newaxis]@np.sum(xx[ind,:], axis=0)[:, np.newaxis].T-\
          		np.sum(xx[ind,:], axis=0)[:, np.newaxis]@m[:,k][:, np.newaxis].T+np.sum(n_vec[ind])*m[:,k][:,np.newaxis]@m[:,k][:,np.newaxis].T)/np.sum(n_vec[ind])+10**(-6)*np.identity(p))*p
        else:
          indx = np.argmax(ind)
          m[:,k]=xm[indx,:]
          D[:,:,k]=((SS[:,:,indx]-m[:,k][:,np.newaxis]@xx[indx,:][:,np.newaxis].T-xx[indx,:][:,np.newaxis]@m[:,k][:,np.newaxis].T+\
          		n_vec[indx]*m[:,k][:,np.newaxis]@m[:,k][:,np.newaxis].T)/n_vec[indx]+10**(-6)*np.identity(p))*p

        phi=np.zeros((n,Kmax))
        phi[:,:k_KM]=phi_KM
    else:
      m=np.zeros((p,Kmax))
      D = np.stack([np.identity(p) for ip in range(Kmax)], axis=2)
      for k in range(n):
        m[:,k]=xm[k,:]
        D[:,:,k]=((SS[:,:,k]-m[:,k][:,np.newaxis]@xx[k,:][:,np.newaxis].T-xx[k,:][:,np.newaxis]@m[:,k][:,np.newaxis].T+\
        		n_vec[k]*m[:,k][:,np.newaxis]@m[:,k][:,np.newaxis].T)/n_vec[k]+10**(-6)*np.identity(p))*p
      phi=np.random.dirichlet(np.tile(1/Kmax,Kmax), n)  
  else:
    m=np.zeros((p,Kmax))
    D = np.stack([np.identity(p) for ip in range(Kmax)], axis=2)
    phi=np.tile(1/Kmax,(n,Kmax))
  xi_est, phi, m, D, labels, elbo, para_diff, Prec =C_VI_PYMMG_CoC(xx,SS,n_vec,max_it,tol,Kmax,theta0,tau0,b0,Omega0,alpha,d,m,D,phi,st)
 
  K_est=labels.shape[0]
  Named = {}
  Named['member_est'] = xi_est
  Named['Prec'] = Prec[:,:,labels]
  Named['mu'] = m[:,labels].T
  return Named
  #return xi_est, K_est, phi[:,labels], m[:,labels], D[:,:,labels], elbo, para_diff


if __name__ == "__main__":      
  print('finish')  
  # SS = np.load('SSv.npy')
  # #print(SS.shape)
  # SS = SS.reshape([2,2,26], order='F')
  # #print(SS[:,:,0])
  # x = np.load('xv.npy')
  # n_vec = np.load('n_vecv.npy')
  # #print(n_vec)
  # xi_estR = np.load('xi_estv.npy')
  # xi_est, K_est, phi, m, D, _, _ = R_VI_PYMMG_CoC(x, SS, n_vec,max_it=200,Kmax=20,tol=0.01)
  # print('Python:{}'.format(xi_est))
  # print('R:{}'.format(xi_estR))
  # print(normalized_mutual_info_score(xi_est, xi_estR))
  # print(K_est)
  # print(phi)
  # print(m)
  # print(D)
