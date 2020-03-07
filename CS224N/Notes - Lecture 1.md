

# Lecture 1: Introduction to Word Vector

How to represent words:

1. Common solution: Use e.g. WordNet, a thesaurus containing lists of **synonym sets** and **hypernyms** (“is a” relationships).

   Problem: 

   ![image-20200307222833851](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307222833851.png)

2. Discrete symbols - One hot encoding

   Problem: 

   * large dimension
   * No **similarity**: Could try to rely on WordNet's list to get similarity but will experience incompleteness

3. Distributional semantics: represent by their context -> word vectors 



**Word2Vec Overview**

![image-20200307223133732](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307223133732.png)

![image-20200307223145415](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307223145415.png)

![image-20200307224744141](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307224744141.png)

Why: max 

We want to acheive knowing what words occur in its context. which means it could accurately give a high probability estimate to those words that occur in the context.

![image-20200307225030284](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307225030284.png)

![image-20200307225701571](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307225701571.png)

We should do some **derivations of gradient**

![image-20200307230030574](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307230030574.png)

![image-20200307230143875](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307230143875.png)

Thus finally we obtain:

![image-20200307230228216](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307230228216.png)

![image-20200307230309901](/Users/zhaoyuji/Desktop/Course Materials/CS224N/img/image-20200307230309901.png)

which is also = observation on this words - weighted average (expectation) of the models of the representations of each word multiply by the probability of it in the current model

= the actual contect word - the expected context word





 