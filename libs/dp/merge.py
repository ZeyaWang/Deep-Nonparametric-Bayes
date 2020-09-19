import numpy as np
from scipy.special import gammaln
#multigammaln
from numpy.linalg import slogdet
import sys
#log multivariate gamma
def lmgamma(p, a):
  C = p * (p - 1) / 4 * np.log(np.pi)
  for i in range(p):
    C += gammaln(a-.5*i)
  return C

#marginal density for Normal-inverse-Wishart
def marg_density(Lambda, kappa, nu, n, p, S, M):
  #Lambda: pxp; S: pxp; M: pxp
  #print(Lambda.shape)
  #print(slogdet(Lambda)[1])
  #print('kappa')
  #print(kappa)
  #print(n)
  res = lmgamma(p,(nu+n)/2)+nu*slogdet(Lambda)[1]/2.+p*np.log(kappa)/2.\
         -n*p*np.log(np.pi)/2-lmgamma(p,nu/2)\
         -(nu+n)*slogdet(Lambda+S+kappa*n*M/(kappa+n))[1]/2.-p*np.log(kappa+n)/2
  #print(res)
  return res


def act_inact(count):
  act = []
  inact = []
  for i in range(len(count)):
    if (count[i]>0):
      act.append(i)
    else:
      inact.append(i)
  return act, inact, len(act)


def member_to_partition(member, ref, Kmax=None):
  #transfer a member vector to a partition binary matrix
  #member: list; ref:list; kmax:int
  n = len(member)
  K = len(ref)
  if Kmax:
    partition = np.zeros((n,Kmax))
  else:
    partition = np.zeros((n,K))
  for i in range(n):
    for k in range(K):
      if (member[i]==ref[k]):
        partition[i,k]=1
  return partition

def normalize(prob):
  #normalize numpy vector summed up to 1
  n = prob.shape[0]
  ## overflow
  probmax = np.max(prob)
  l = np.log(np.sum(np.exp(prob-probmax))) + probmax
  result = np.exp(prob-l)
  return result/np.sum(result)

def CoClusterProb(member, n_iter, n):
  co = np.zeros((n,n)) #pairwise co-clustering probabilities
  for iter in range(n_iter):
    for i in range(n):
      for j in range(i):
        if (member[j,iter] == member[i,iter]):
          co[i,j] += 1
  co=co/n_iter
  return co


def LScluster(member, n_iter, n, co):
  minDB = sys.float_info.max
  for iter in range(n_iter):
    db = 0.0
    for i in range(n):
      for j in range(i):
        tmp = (1 if member[i,iter] == member[j,iter] else 0) - co[i,j]
        db += tmp*tmp
    if db < minDB: 
      minDB = db
      minIndex = iter
  return minIndex


def Gibbs_DPM_Gaussian_summary_input(SS, x, n_vec, 
                                     kappa0=0.01, alpha=1, n_iter=10):

  '''
  //INPUT:
  //SS: p x p x K array of sum-of-square matrices; each slice of the array is a p by p matrix (x'x within each cluster) and the number of slices is the number of input clusters
  xTx
  //x: K x p matrix; each row is sum of features over samples within each cluster
  //n_vec: K-dim vector; cluster size
  //nu0=0.01: degrees_of_freedom_prior (python, BayesianGaussianMixture) but I would set nu0=0.01 rather than the default choice 1 in BayesianGaussianMixture
  //kappa0=0.01: similar to mean_precision_prior (python, BayesianGaussianMixture)
  //Lambda0=I: covariance_prior (python, BayesianGaussianMixture)
  //alpha=1: weight_concentration_prior (python, BayesianGaussianMixture)
  //d=0;
  //note: mean_prior in BayesianGaussianMixture is set to mean of X by default, in my code, it's 0.
  //n_discard=0 (burnin)
  //thin=1 (thinning)
  #nu0=0.01
  #kappa0=0.01
  #Lambda0=diag(p)
  #member_ini=0:(n-1)
  #count_ini=rep(1,Kmax)
  #n_iter=20
  '''
  Lambda0=np.identity(SS.shape[0]) 
  member_ini=list(range(SS.shape[2]))
  count_ini=np.copy(n_vec)
  p, _, n = SS.shape
  nu0 = p
  alpha=1
  Kmax=n
  effMCsize=n_iter
  member = np.zeros((n,effMCsize))
  count = np.zeros((n,effMCsize))
  K = np.ones(effMCsize)
  member_old = member_ini.copy() #membership of each samples; list
  count_old = count_ini.copy() #size of each cluster
  
  
  act,inact,K_old=act_inact(count_old)
  partition=member_to_partition(member_old,act,Kmax) #binary matrix partition(i,j)=1 if ith sample in jth cluster
  #partition is numpy mat
  iter_save=0
  #mat S(p,p); #mat M(p,p);
  Lambdan,kappan,nun=np.copy(Lambda0),kappa0,nu0
  print("Gibbs starts\n")
  for iter in range(n_iter):
    #update cluster assignment
    #print('n_vec{}'.format(n_vec))
    for i in range(n):
      prob = np.zeros(min(K_old+1,Kmax))
      prob.fill(-10.**10)
      for kk in range(K_old):
        k=act[kk]
        if (member_old[i]==k):
          if (count_old[k] > n_vec[i]): #if not the last element in cluster k
            part_tmp=partition[:,k]
            part_tmp[i] = 0
            rowind=np.nonzero(part_tmp)[0] #find nonzero index
            xtmp = np.sum(x[rowind,:], axis=0) #Sums over row
            M=np.matmul(xtmp[np.newaxis,:].T,xtmp[np.newaxis,:])
            S=np.sum(SS[:,:,rowind],axis=2) #sum over the last dimesion of 3d array
            S=S-M/(count_old[k]-n_vec[i])
            M=M/(count_old[k]-n_vec[i])/(count_old[k]-n_vec[i])
            Lambdan=Lambda0+S+kappa0*(count_old[k]-n_vec[i])*M/(kappa0+count_old[k]-n_vec[i])
            kappan=kappa0+count_old[k]-n_vec[i]
            nun=nu0+count_old[k]-n_vec[i]
            mun=xtmp/kappan
            M=np.matmul(x[i,:][np.newaxis,:].T, x[i,:][np.newaxis,:])
            S=SS[:,:,i]-M/n_vec[i] #SS.slice(i)=SS[,,i], SS.slices(ind)=SS[,,ind]
            M=M/n_vec[i]/n_vec[i]-np.matmul(x[i,:][np.newaxis,:].T,mun[np.newaxis,:])/n_vec[i]-np.matmul(mun[np.newaxis,:].T, x[i,:][np.newaxis,:])/n_vec[i]+np.matmul(mun[np.newaxis,:].T, mun[np.newaxis,:])
            prob[kk]=gammaln(count_old[k])-gammaln(count_old[k]-n_vec[i])+marg_density(Lambdan, kappan, nun, n_vec[i], p,  S,  M)
        else:
          rowind=np.nonzero(partition[:,k])[0]
          xtmp = np.sum(x[rowind,:], axis=0) #Sums over row
          M=np.matmul(xtmp[np.newaxis,:].T,xtmp[np.newaxis,:])
          S=np.sum(SS[:,:,rowind],axis=2) #sum over the last dimesion of 3d array
          S=S-M/count_old[k]
          M=M/count_old[k]/count_old[k]
          Lambdan=Lambda0+S+kappa0*count_old[k]*M/(kappa0+count_old[k])
          kappan=kappa0+count_old[k]
          nun=nu0+count_old[k]
          mun=xtmp/kappan
          M=np.matmul(x[i,:][np.newaxis,:].T, x[i,:][np.newaxis,:])
          S=SS[:,:,i]-M/n_vec[i] #SS.slice(i)=SS[,,i], SS.slices(ind)=SS[,,ind]
          #print('1st n_vec{}'.format(n_vec))
          #print('1st count_old{}'.format(count_old))

          M=M/n_vec[i]/n_vec[i]-np.matmul(x[i,:][np.newaxis,:].T,mun[np.newaxis,:])/n_vec[i]-np.matmul(mun[np.newaxis,:].T, x[i,:][np.newaxis,:])/n_vec[i]+np.matmul(mun[np.newaxis,:].T, mun[np.newaxis,:])
          ##
          #print('2nd n_vec{}'.format(n_vec))
          #print('2nd count_old{}'.format(count_old))
          prob[kk]=gammaln(count_old[k]+n_vec[i])-gammaln(count_old[k])+marg_density(Lambdan, kappan,  nun, n_vec[i], p,  S,  M)
        
      if (K_old<Kmax):
        M=np.matmul(x[i,:][np.newaxis,:].T, x[i,:][np.newaxis,:])
        S=SS[:,:,i]-M/n_vec[i]
        M=M/n_vec[i]/n_vec[i]
        #
        prob[K_old]=np.log(alpha)+gammaln(n_vec[i])+marg_density(Lambda0, kappa0,  nu0, n_vec[i], p,  S,  M)
      
      prob=normalize(prob) #normalize probability. In R: prob/sum(prob)
      multinom_vector = np.random.multinomial(1, prob) #random sample from multinomial
      cluster_this= np.where(multinom_vector==1)[0][0] #cluster label
      
      if (cluster_this<K_old):
        s_tmp=act[cluster_this]
      else:
        s_tmp=inact[0]
      
      #adjust the partitions and their sizes
      if (cluster_this==K_old or s_tmp!=member_old[i]):
        #print(n_vec)
        count_old[s_tmp]=count_old[s_tmp]+n_vec[i]
        count_old[member_old[i]]=count_old[member_old[i]]-n_vec[i]
        #print(count_old)
        partition[i,s_tmp]=1
        partition[i,member_old[i]]=0
      
      member_old[i]=s_tmp
      #      
      act, inact, K_old = act_inact(count_old)
      

    member[:,iter_save]=member_old
    count[:,iter_save]=count_old
    K[iter_save]=K_old
    iter_save += 1
  
  print("Gibbs ends; start summarizing\n")
  #Summarize clustering: Dahl's LS algorithm
  co = CoClusterProb(member,effMCsize,n)
  minIndex=LScluster(member,effMCsize,n,co)
  member_est=member[:,minIndex]
  count_est=count[:,minIndex]
  act, _, K_est=act_inact(count_est)
  partition=member_to_partition(member_est,act)
  
  mu = np.zeros((K_est,p))
  Sig = np.zeros((p,p,K_est))
  Prec = np.zeros((p,p,K_est))
  for kk in range(K_est):
    k = act[kk]
    rowind=np.nonzero(partition[:,kk])[0]
    mu[kk,:]=np.sum(x[rowind,:]/(kappa0+count_est[k]), axis=0) # sum over row
    xtmp = np.sum(x[rowind,:], axis=0)
    M=np.matmul(xtmp[np.newaxis,:].T, xtmp[np.newaxis,:])
    S=np.sum(SS[:,:,rowind],axis=2)
    S=S-M/count_est[k]
    M=M/count_est[k]/count_est[k]
    Lambdan=Lambda0+S+kappa0*count_est[k]*M/(kappa0+count_est[k])
    if (nu0+count_est[k]-p-1)>0:
      Sig[:,:,kk]=Lambdan/(nu0+count_est[k]-p-1)
      Prec[:,:,kk]=np.linalg.inv(Lambdan/(nu0+count_est[k]-p-1))
    else:
      Sig[:,:,kk]=Lambdan/(nu0+count_est[k]-p)
      Prec[:,:,kk]=np.linalg.inv(Lambdan/(nu0+count_est[k]-p))
    
  Named = {}
  Named["member_est"] = member_est
  Named["member"] = member
  Named["count"] = count_est
  Named["partition"] = partition
  Named["K_est"] = K_est
  Named["K"]=K
  Named["Co"]=co
  Named["Sig"]=Sig
  Named["Prec"]=Prec
  Named['mu']=mu
  return Named

if __name__ == "__main__":        
  SS = np.load('SS.npy').reshape([5,5,19], order='F')
  #print(SS[:,:,0])
  x = np.load('x.npy')
  n_vec = np.load('n_vec.npy')
  Rest = Gibbs_DPM_Gaussian_summary_input(SS, x, n_vec)
  #print(Rest["K_est"])
  #print(Rest["K"])
  #print(Rest['mu'])
  #print(Rest['Sig'])
  print(Rest['member_est']) 
  #print(Rest["count"])
