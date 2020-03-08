# Lecture 2: Word Vector

### Review

![image-20200308202922762](img/image-20200308202922762.png)

Suppose for one word i it has two vectors $v_i$ and $u_i$. When you use dot product between $U$ and $v_4$, it results in $[u_1v_4^T, u_2v_4^T, \cdots, u_Nv_4^T]$. Then after you apply softmax function, you could get:

![image-20200308203502629](img/image-20200308203502629.png) 

### How to train parameters:

![image-20200308100724696](img/image-20200308100724696.png)

In this setting, we assume there are two vectors for one word, but actually we could also use one vector for one word.

Currently there are two main models:

![image-20200308212701256](img/image-20200308212701256.png)

**The skip-gram model with negative sampling**

Replace all words set V with some negative samples (random words)

![image-20200308212844566](img/image-20200308212844566.png)



###  **co-occurrence counts** 

![image-20200308212937638](img/image-20200308212937638.png)

![image-20200308213016957](img/image-20200308213016957.png)

#### Solutions:**Low dimensional vectors**

**Methods1:** 

Dimensionality Reduction on X (HW1) -> Singular Value Decomposition of co-occurrence matrix *X*

**Tips:**  Scaling the counts in the cells can help **a lot**

1. Set some max/min limitations for words

2. Ramped windows that count closer words more

3. Use Pearson correlations instead of counts, then set

   negative values to 0

![image-20200308213344547](img/image-20200308213344547.png)



### Other Variations: GloVe  [2014]

Encoding meaning in vector differences

![image-20200308213506240](img/image-20200308213506240.png)

![image-20200308213806357](img/image-20200308213806357.png)

**Advs:**

* Fast training

* Scalable to huge corpora

* Good performance even with small corpus and small vectors

### **How to evaluate word vectors?**

![image-20200308213944996](img/image-20200308213944996.png)



### CAP:

How to learning word embeddings:

1. Full document: LSA and etc.

2. Window size Information: CBOW (predict center word by words around) and Skip Gram (predict words around by center word)

3. GloVe considers two part above thus it could ignore those words whose co-occurence is 0 (which matrix factorization could not avoid) to reduce memory usage and calculation, and its learning method is similar to CBOW and Skip Gram which could be easily solved by optimization methods.

   

一点心得：

其实上第一节课的时候老师讲词向量的时候用了一个analogy的例子就让我很为之惊叹，其实只是一个很细微的差别。之前大多听的课老师举词向量的时候都会用这样类似的例子，比如“上海”和“中国”两个点连起来的线可能和“纽约”和“美国”连起来的线平行，表示他们比较类似，而这节课上manning举出的例子却有些不太一样（下图），他给出的这个例子是：king-man + woman -> queen. 虽然这个例子本质上和上面的通用例子很相似，但它感觉上更能揭示出词向量在表示semantic similarity上的优点，以及颇有些知识推理的味道在里面

![image-20200308211032149](img/image-20200308211032149.png)

第二个感触很深的点在于写assignment1的时候，其中有一题是这样的：

![image-20200308211311894](img/image-20200308211311894.png)

这是在我之前的课程、项目中从未被提及或者接触到的一点！考虑word embedding 会不会存在sexual racial or gender biases！这可能就是很不一样的 vision吧。。