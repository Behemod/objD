# Some math to understand nn better

## Math as nn goes:

### &emsp;**Propagations:**
- &emsp;&emsp;Forward propagation:
    $$Z_n=W_n*A_{n-1}+b_n$$
    $$A_n=g_{any}(Z_n)$$
    Where $n$ is number of hidden layer, $W_n$ is weight vector, $b_n$ is bias vector, $Z_n$ is linear combination of $A_{n-1}$

- &emsp;&emsp;Backward propagation:
    $$$$


### &emsp;**Activation:**

- &emsp;&emsp;Sigmoid: $g_{sigmoid}(x)={1 \over 1 + e^{-x}}$
- &emsp;&emsp;Rectified Linear Unit (ReLU): $g_{ReLU}(x)=\left[\begin{aligned}&x=0,~x\le0\\&x=x,~x\gt0\end{aligned}\right.$
- &emsp;&emsp;Softmax: $g_{softmax}(x)={e^x \over \sum e^x}$

&emsp; Makes all thing non-linear


### &emsp;**Optimization:**