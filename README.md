# LogisticRegression

The logisticRegression.cpp File contains the model. The data.cpp files contains a function to read in the Wisconsin Breat-Cancer Dataset (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). The logistic Regression had a accuracy of 90-95% dependign on the train-test-split.

## Generalized Linear Models

We try to train our model to predict a target value $y$ given the input features $\vec{x}$. When we have multiple Sets of inputs and outputs I describe the features as $X$ and the targets as $\vec{y}$.  
Since our targets are either 0 or 1, we can use the Bernoulli-distrubution. Because this is in the exponantial Family, we can apply a Generalized Linear Model. We try to adjust the parameters $\theta$ such that $h_\theta(x)= E[y|x;\theta]$ is satisfied. Additional we assume that $y|x;\theta \sim ExponantialFamily(\eta)$ where the Distrubution from the ExponantialFamily can be written as $p(y; \eta)=b(y)exp(\eta^TT(y)-a(\eta))$ and that $\eta$ and $x$ are related linearly ($\eta=\theta^Tx$).

For the Bernoulli-distrubution holds:  
$p(y; \phi)=\phi^y * (1-\phi)^{1-y}$  
$p(y;\phi)=\exp(y \log {\phi} + (1-y) \log {(1-\phi)})$  
$p(y;\phi)=\exp(y \log {\phi} + -y \log {(1-\phi)} + \log {(1-\phi)})$  
$p(y;\phi)=\exp(y \log {\frac{\phi}{1-\phi}} + \log {(1-\phi)})$  


From this we can see  
$b(y)=1$  
$\eta=\log {\frac{\phi}{1-\phi}} \Leftrightarrow \phi = \frac{1}{1-\exp(\eta)}$  
$T(y)=y$  
$a(\eta)=-\log {(1-\phi)} = \log {(1+\exp(\eta))}$

Now $h_\theta(x)$ follows from $h_\theta(x)= E[y|x;\theta]$.  
$h_\theta(x)= E[y|x;\theta]=\phi=\frac{1}{1-\exp(\eta)}$  
$h_\theta(x)=\frac{1}{1-\exp(\theta^Tx)} = \frac{1}{1+\exp(-\theta^Tx)}$

$g(z) = \frac{1}{1+\exp(z)}$ is the sigmoid function, which has the property $g'(z) = g(z) * (1-g(z))$.  

To adjust the parameters $\theta$ we can maximize the Likelihood $L(\theta)=p(\vec{y}|X;\theta)=\prod_{i=1}^{m} p(y^{(i)}|x^{(i)};\theta)$ with gradient descent.  
Since the $log$-function is strictly increasing we can also maximize this, to make the algebra a bit easier.  
$l(\theta)=\log(L(\theta))=\sum_{i=1}^{m} \log(p(y^{(i)}|x^{(i)};\theta))$  
$l(\theta)=\sum_{i=1}^{m} \log((\frac{1}{1+\exp(-\theta^Tx^{i})})^{y^{i}} * (1-\frac{1}{1+\exp(-\theta^Tx^{i})})^{1-y^{i}}) = \sum_{i=1}^{m} y^{i}* \log(g(\theta^Tx^{i})) + \(1-y^{i}) * \log(1-g(\theta^Tx^{i}))$  

So $l(\theta)$ is our Costfunction $J(\theta)$, which we are trying to minimze.

Gradient Descent works by adjusting the paramters in the steepest direction. The update-rule says $\theta_j \leftarrow \theta_j + \alpha \frac{\partial}{\partial \theta_j} J(\theta)$ with $\alpha$ being the learning rate.  

I will derive the Update-rule with a single training example and then modify it for multiple.  
$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{2} (y-\theta^Tx)^2 = \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_\theta(x)-y)^2$  
$\frac{\partial}{\partial \theta_j} J(\theta) = (h_\theta(x)-y) * \frac{\partial}{\partial \theta_j} (h_\theta(x)-y) = (h_\theta(x)-y) * \frac{\partial}{\partial \theta_j} (\sum_{i=0}\theta_i^Tx_i -y)$  
$\frac{\partial}{\partial \theta_j} J(\theta) = (h_\theta(x)-y)x_j$

For multiple training examples we can use either Batch Gradient Descent (BGD), by looking on every example for one step or we can use Stochastic Gradient Ascent (SGD) by adjusting the paremeters for every example.  
BGD: $\theta_j \leftarrow \theta_j + \alpha \sum_{i=1}^{m} (h_\theta(x^{(i)})-y^{(i)})x_j$ (for every $j$)  
SGD: for $i=1$ to $m$ { $\theta_j \leftarrow \theta_j + \alpha (h_\theta(x^{(i)})-y^{(i)})x_j$ (for every $j$) }


(Source: https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
