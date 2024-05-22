
[paper](https://www.ijcai.org/Proceedings/07/Papers/416.pdf)

# Introduction
- IRL Problem:
	- **Determine** The reward function that an agent is optimizing
	- **Given**
		- Measurement of the agent's behaviour over time, in a variety of circumstances;
		- Measurements of the sensory inputs to that agent;
		- A model of the environment. In the context of Markov Decision Processes, this translates into determining the reward function of the agent from knowledge of the policy it executes and dynamic of the sate-space.

There are two tasks that IRL accomplishes.
- *Reward Learning*: estimating the unknown reward function.
- *Apprenticeship learning*: using observations of an expert's actions to decide one's own behaviour.

This paper considers the IRl problem from a Bayesian perspective. The expert actions are cosidered as *evidence* that's used to update a prior on reward functions.

# Preliminaries
#### Markov Decision Problem
Is a tuple $(S,A,T,\gamma,R)$ where: 
- S is a finite set of N *states*.
- $A=\{a_1,...,a_k\}$ is a set of k *actions*.
- $T:S\times A\times S\mapsto[0,1]$ is a *transition probability function*.
- $\gamma\in[0,1)$ is the *discount factor*.
- $R:S\mapsto \mathbb{R}$ is a *reward function*, with absolute values bounded by $R_{max}$ 
The rewards are functions of state alone because IRL problems typically have limited information about the value of action and we want to avoid overfitting.
#### Markov Decision Process (MDP)
is a tuple $(S,A,T,\gamma)$ with the terms defined as before but without a reward function. To avoid cunfusion the abbreviation MDP will be used only for Markov Decision Processes and not Problems.

The following compact notation will be adopted for finite MDPs: Fix an enumeration $s_1...s_N$ of the finite state space $S$. The reward function can then be represented as an N-dimensional vector **R**, whose i-th element is $R(s_i)$.

A (stationary) *policy* is a map $\pi:S\mapsto A$ and the *value* of a policy $\pi$ for reward function **R** at state $s\in S$, denoted $V^\pi(s,\mathbf R)$ is given by:
$$V^\pi(s_{t_1},\mathbf R) = R(s_{t_1})+E_{s_{t_1},s_{t_2,...}}[\gamma R(s_{t_2})+\gamma^2R(s_{t_3})+...|\pi]$$
Where $Pr(s_{t_{i+1}}|s_{t_i},\pi)=T(s_{t_i},\pi(s_{t_i}),s_{t_{i+1}}))$. The goal of standard Reinforcement Learning is to find an *optimal policy* $\pi^*$ such that $V^\pi(s,\mathbf R)$  is maximized for all $s\in S$ by $\pi=\pi^*$. It can be shown that at least one such policy always exists for ergodic MDPs.

For the solution of Markov Decision Problems, it's useful to define the following auxilliary $Q-function$:
$$Q^\pi(s,a,\mathbf R)=R(s)+ \gamma E_{s'\sim T(s,a,\cdot)}[V^\pi(s',\mathbf R)]$$ 
#### Theorem 1 (Bellman Equations)
Let a Markov Decision Problem $M=(S,A,T,\gamma,R)$ and a policy $\pi:S\mapsto A$ be given. Then,
- For all $s\in S,a\in A,V^\pi$ and $Q^\pi$ satisfy
$$V^\pi(s)=R(s)+\gamma\sum_{s'}T(s,\pi(s),s')V^\pi(s')$$$$Q^\pi(s,a)=R(s)+\gamma\sum_{s'}T(s,a,s')V^\pi(s')$$
- $\pi$ is an optimal policy for M iff, for all $s\in S$,
$$\pi(s)\in \text{argmax}_{a\in A}Q^\pi(s,a)$$

# Bayesian IRL

IRL is currently viewed (at time of paper) as a problem of infering a single reward function that explains an agent's behaviour. However, there is too little information in a typical IRl problem to get only one answer.  Thus, a probability distribution is needed to represent the uncertainty (see example in paper).

## Evidence from the Expert
In the Bayesian IRL model, we derive a posterior distribution for the rewards from a prior distribution and a probabilistic model of the expert's actions given the reward function.
Consider an MDP M and an agent $\chi$ (the expert) operating in this MDP. We assume that a reward function **R** for $\chi$ is chosen from a (known) prior distribution $P_R$. The IRL agent receives a series of observations of the expert's behaviour $O_\chi=\{(s_1,a_1),...,(s_k,a,_k)\}$ which means that $\chi$ was in state $s_i$ and took action $a_i$ at that step i. For generality, we will not specify the algorithm that $\chi$ uses to determine his (possibly stochastic policy, but we make the following assumptions about his behaviour:
- $\chi$ is attempting to maximize the total accumulated reward according to **R**. For example, $\chi$ is not using a greedy policy to explore his environment.
- $\chi$ executes a stationary policy, it's invariant w.r.t. time and does not change depending on the actions and observations made in previous time steps.
Because the  expert's policy is stationary, we can make the following independence assumption:
$$P_\chi(O_\chi|\mathbf R)=P_\chi((s_1,a_1)|\mathbf R)...P_\chi((x_k,a_k)|\mathbf R)$$
The expert's goal of maximizing accumulated reward is equivalent to finding the action for which the $Q^*$ value at each state is maximum. Therefore, the larger $Q^*(s,a)$ is, the more likely it is that $\chi$ would choose action $a$ at time $s$. This likelihood increases the more confident we are in $\chi$'s ability to select a good action. We model this by an exponential distribution for the likelihood of $(s_i,a_i)$, with $Q^*$ as a potential function:
$$P_\chi((s_i,a_i)|\mathbf R)=\frac{1}{Z_i}e^{\alpha\chi Q^*(s_i,a_i,\mathbf R)}$$
where $\alpha_\chi$ is paremeter representing the degree of confidence we have in $\chi$'s ability to choose actions with high values.
The likelihood of the entire evidence is:
$$P_\chi(O_\chi|\mathbf R)=\frac{1}{Z}e^{\alpha_\chi E(O_\chi,\mathbf R)}$$
Where $E(O_\chi,\mathbf R)=\sum_i Q^*(s_i,a_i,\mathbf R)$ and Z is the appropriate normalizing constant.
Now we compute the posterior probability of reward function $\mathbf R$ by applying Bayes theorem,
$$P_\chi(\mathbf R|O_\chi)=\frac{P_\chi(O_\chi|\mathbf R)P_R(\mathbf R)}{P(O_\chi)}$$
$$P_\chi(\mathbf R|O_\chi)=\frac{1}{Z_i}e^{\alpha_\chi E(O_\chi,\mathbf R)}P(\mathbf R)$$
## Priors
When no other information is given, we may assume that the rewards are i.i.d. by the principle of maximum entropy. Most of the prior functions considered in this paper will be of this form. The exact prior to use however, depends on the characteristics of the problem:
- If we are completely agnostic about the prior, we can use the uniform distribution over the space $-R_{max}\le R(s)\le R_{max}$ for each $s\in S$. If we do not want to specify any $R_{max}$ we can try the improper prior $P(\mathbf R)=1$ for all $\mathbf R\in \mathbb R^n$ 
- Many real world Markov decision problems have parsimonious reward structures, with most states having negligible rewards. In such situations, it would be better to assume a Gaussian or Laplacian prior.
- If the underlying MDP represented a planning-type problem, we expect most states to have low (or negative) rewards but a few state to have high rewards (corresponding to the goal); this can be modeled by a $Beta$ distribution for the reward at each state, which has mode at high and low ends of the reward space.

# Inference
Our general procedure is to derive minimal solutions for appropriate loss functions over the posterior.

## Reward Learning
Reward learning is an estimation task. The mos common loss functions are:
$$L_{linear}=(\mathbf R,\mathbf{\hat R})=||\mathbf R-\mathbf{\hat R}||_1$$
$$L_{SE}(\mathbf R,\mathbf{\hat R}) = ||\mathbf R,\mathbf{\hat R}||_2$$
Where $\mathbf R$ and $\mathbf{\hat R}$ are the actual and estimated rewards, respectively.

## Apprenticeship Learning
For the apprenticeship learning task the situation is more interesting. Since we are attempting to learn a policy $\pi$, we can formally define the following class of *policy loss functions*:
$$L_{policy}^p(\mathbf R,\pi)=||\mathbf{V^*-(R)-V^\pi(R)}||_p$$
where $\mathbf V^*(\mathbf R)$ is the vector of optimal values for each state achieved by the optimal policy for **R** and p is some norm.

- **Theorem** *Given a distribution $P(\mathbf R)$ over reward functions $\mathbf R$ for an MDP, the loss function$L_{policy}^p(\mathbf R,\pi)$ is _minimizes for all p by $\pi^*_M$*, the optimal policy for the Markov decision problem $M=(S,A,T,\gamma,E_P[\mathbf R])$.

# Sampling and Rapid Convergence
The sampling technique used in the paper is an MCMC algorithm GridWalk that generates a Markov chain on the intersection points of a grid of length $\delta$ in the region $\mathbb R^{|S|}$ (denoted $\mathbb R^{|S|}/\delta$).
The paper proposes a modified version of GridWalk called PolicyWalk: while moving along a Markov chain, the sampler also keeps track of the optimal policy $\pi$ for the current reward vector $\mathbf R$. Observe that when $\pi$ is known, the $Q$ function can be reduced to a linear function of the reward variables. A change in the optimal policy can easily be detected when moving to the next reward vector in the chain $\mathbf{\tilde{R}}$, because then for some $(s,a) \in (S,A)$, $Q^\pi(s,\pi(s),\mathbf{\tilde R})<Q^\pi(s,a,\mathbf{\tilde R})$    by theorem 1. When this happens, the new optimal policy is usually only slightly different from the old one and can be computed by just a few steps of policy iteration starting from the old policy $\pi$. Hence, PolicyWalk is a correct and efficient sampling procedure.

- **Lemma**. *Let $F(\cdot)$ be a positive real valued function defined on the cube $\{x\in\mathbb R^n|-d\le x_i\le d\}$ for some positive d, satisfying for all $\lambda \in [0,1]$ and some $\alpha,\beta$*
$$|f(x)-f(y)|\le\alpha||x-y||_\infty$$
and
$$f(\lambda x+(1-\lambda)y)\ge\lambda f(x)+(1-\lambda)f(y)-\beta$$
*where $f(x)=\log F(x)$. Then the markov chain induced by GridWalk (and hence PolicyWalk) on F rapidly mixes to within $\epsilon$ of $F$ in $O(n²d²\alpha²e^{2\beta}\log{\frac 1 \epsilon})$ steps.*
- **Theorem** *Given an MDP $M=(S,A,T\gamma)$ with $|S|=N$, and a distribution over rewards $P(\mathbf R)=Pr_\chi(\mathbf R|O_\chi)$ with uniform prior. If $R_{max}=O(1/N)$ then P can be efficiently sampled (within error $\epsilon) in $O(N^2\log{1/\epsilon})$ steps by algorithm PolicyWalk.*
Note that having $R_{max}=O(1/N)$ is not really a restriction because we  can rescale the rewards by a constant factor k after computing the mean without changing the optimal policy and all the value functions and $Q$ functions get scaled by $k$ as well.

# Experiments
We compared the performance of our BIRL aproach to the IRL algorithm of \[Ng and Russell, 200\] experimentally. 
First, we generated random MDPs with N states (with N varying from 10 to 1000) and rewards drawn from i.i.d Gaussian priors. Then, we simulated two kinds of agents on these MDPs and used their trajectories as input: The first learned a policy by Q-learning on the MDP + reward function. The learning rate was controlled so that the agent was not allowd to converge to the optimal policy but came reasonably close. The second agent executed a policy that maximizes the expected total reward over the next k steps (k was chosen to be slightly below the horizon time). 
For BIRL, we used PolicyWalk to sample the posterior distributionwith a uniform prior. We campared the results of the two methods by their average $\mathcal l_2$ distance from the true reward function and the policy loss with $\mathcal l_1$ norm of the learned policy under the true reward.
We also measured the accuracy of our posterior distribution for small N by comparing it with the true distribution of rewards i.e. the set of generated rewards that gave rise to the same trajectory by the expert. 
## From Domain Knowledge to Prior
To show how domain knowlege about a problem can be incorporate into the IRL formulation as an informative prior, we applied our methods to learning reward functions in adventure games. There, an agent explores a dungeon, seeking to collect various items of treasure and avoid obstacles such as guards or traps. The state is represented by an m-dimensional binary feature vector indicating the position of the agent and the value of various fluents such as hasKey and doorLocked. If we view the state-space as m-dimensional lattice $L_S$, we see that nighbouring states in $L_S$ are likely to have correlated rewars. To model this, we use an Ising prior.
We tested our hypothesis by generating some adventure games and testing the performance of BIRL with the Ising prior vesus the baseline uninformed prior.