# Translation: Deep packet

<!-- TOC -->

- [Translation: Deep packet](#translation-deep-packet)
  - [Deep packet: a novel approach for encrypted traffic classification using deep learning](#deep-packet-a-novel-approach-for-encrypted-traffic-classification-using-deep-learning)
    - [Author](#author)
    - [Abstract](#abstract)
    - [Keywords](#keywords)
    - [1 Introduction](#1-introduction)
    - [2 related works](#2-related-works)
    - [3 Background on deep neural networks](#3-background-on-deep-neural-networks)
      - [3.1 Autoencoder](#31-autoencoder)
      - [3.2 Convolutional neural network](#32-convolutional-neural-network)
    - [4 Methodology](#4-methodology)
      - [4.1 Dataset](#41-dataset)
      - [4.2 Pre-processing](#42-pre-processing)
        - [4.2.1 Labeling dataset](#421-labeling-dataset)
      - [4.3 Architectures](#43-architectures)
    - [5 Experimental results](#5-experimental-results)
      - [5.1 Comparison](#51-comparison)
        - [5.1.1 Comparison with previous results](#511-comparison-with-previous-results)
        - [5.1.2 Comparison with previous methods](#512-comparison-with-previous-methods)
    - [6 Discussion](#6-discussion)
    - [7 Future work](#7-future-work)
    - [8 Conclusion](#8-conclusion)
  - [深度包：一种使用深度学习进行加密流量分类的新方法](#深度包一种使用深度学习进行加密流量分类的新方法)
    - [作者](#作者)
    - [摘要](#摘要)
    - [关键字](#关键字)
    - [1 简介](#1-简介)
    - [2 相关工作](#2-相关工作)
    - [3 深度神经网络背景](#3-深度神经网络背景)
      - [3.1 自动编码器](#31-自动编码器)
      - [3.2 卷积神经网络](#32-卷积神经网络)
    - [4 方法论](#4-方法论)
      - [4.1 数据集](#41-数据集)
      - [4.2 预处理](#42-预处理)
        - [4.2.1 标注数据集](#421-标注数据集)
      - [4.3 架构](#43-架构)
    - [5 实验结果](#5-实验结果)
      - [5.1 比较](#51-比较)
        - [5.1.1 与以往结果的比较](#511-与以往结果的比较)
        - [5.1.2 与以往方法的比较](#512-与以往方法的比较)
    - [6 讨论](#6-讨论)
    - [7 未来的工作](#7-未来的工作)
    - [8 结论](#8-结论)

<!-- /TOC -->

## Deep packet: a novel approach for encrypted traffic classification using deep learning

### Author

Mohammad Lotfollahi1 · Mahdi Jafari Siavoshani1 · Ramin Shirali Hossein Zade1 · Mohammdsadegh Saberian1

### Abstract

Network traffic classification has become more important with the rapid growth of Internet and online applications. Numerous studies have been done on this topic which have led to many different approaches. Most of these approaches use predefined features extracted by an expert in order to classify network traffic. In contrast, in this study, we propose a deep learning-based approach which integrates both feature extraction and classification phases into one system. Our proposed scheme, called “Deep Packet,” can handle both traffic characterization in which the network traffic is categorized into major classes (e.g., FTP and P2P) and application identification in which identifying end-user applications (e.g., BitTorrent and Skype) is desired. Contrary to most of the current methods, Deep Packet can identify encrypted traffic and also distinguishes between VPN and non-VPN network traffic. The Deep Packet framework employs two deep neural network structures, namely stacked autoencoder (SAE) and convolution neural network (CNN) in order to classify network traffic. Our experiments show that the best result is achieved when Deep Packet uses CNN as its classification model where it achieves recall of 0.98 in application identification task and 0.94 in traffic categorization task. To the best of our knowledge, Deep Packet outperforms all of the proposed classification methods on UNB ISCX VPN-nonVPN dataset.

### Keywords

Network traffic classification · Application identification · Traffic characterization · Deep learning · Convolutional neural networks · Stacked autoencoder · Deep Packet

### 1 Introduction

Traffic classification is an important task in modern communication networks (Bagui et al. 2017). Due to the rapid growth of high-throughput traffic demands, to properly manage network resources, it is vital to recognize different types of applications utilizing network resources. Consequently, accurate traffic classification has become one of the prerequisites for advanced network management tasks such as providing appropriate Quality-of-Service (QoS), anomaly detection, pricing, etc. Traffic classification has attracted a lot of interests in both academia and industrial activities related to network management (e.g., see Dainotti et al. 2012;Finsterbusch et al. 2014; Velan et al. 2015) and the references therein).

As an example of the importance of network traffic classification, one can think of the asymmetric architecture of today’s network access links, which has been designed based on the assumption that clients download more than what they upload. However, the pervasiveness of symmetric-demand applications [such as peer-to-peer (P2P) applications, voice over IP (VoIP) and video call] has changed the clients’ demands to deviate from the assumption mentioned earlier. Thus, to provide a satisfactory experience for the clients, an application-level knowledge is required to allocate adequate resources to such applications.

The emergence of new applications as well as interactions between various components on the Internet has dramatically increased the complexity and diversity of this network which makes the traffic classification a difficult problem per se. In the following, we discuss in details some of the most critical challenges of network traffic classification.

First, the increasing demand for user’s privacy and data encryption has tremendously raised the amount of encrypted traffic in today’s Internet (Velan et al. 2015). Encryption procedure turns the original data into a pseudo-random-like format with the aim to make it hard to decrypt. As a result, it causes the encrypted data scarcely contain any discriminative patterns to identify network traffic. Therefore, accurate classification of encrypted traffic has become a real challenge in modern networks (Dainotti et al. 2012).

It is also worth mentioning that many of the proposed network traffic classification approaches, such as payload inspection as well as machine learning-based and statistical-based methods, require patterns or features to be extracted by experts. This process is prone to error, time-consuming and costly.

Finally, many of the Internet service providers (ISPs) block P2P file sharing applications because of their high bandwidth consumption and copyright issues (Lv et al. 2014). Hence, to circumvent this problem, these applications use protocol embedding and obfuscation techniques to bypass traffic control systems (Alshammari and Zincir-Heywood 2011). The identification of this kind of applications is one of the most challenging tasks in network traffic classification.

There have been abundant studies on the network traffic classification subject, e.g., Kohout and Pevný (2018), Perera et al. (2017),Giletal.(2016) and Moore and Papagiannaki (2005). However, most of them have focused on classifying a protocol family, also known as traffic characterization (e.g., streaming, chat, P2P, etc.), instead of identifying a single application, which is known as application identification (e.g., Spotify, Hangouts, BitTorrent, etc.) (Khalife et al. 2014). In contrast, this work proposes a method, i.e., Deep Packet, based on the ideas recently developed in the machine learning community, namely deep learning (Bengio 2009; LeCun et al. 2015), to both characterize and identify the network traffic. The benefits of our proposed method, which make it superior to other classification schemes, are stated as follows:

- In Deep Packet, there is no need for an expert to extract features related to network traffic. In light of this approach, the cumbersome step of finding and extracting distinguishing features has been omitted.
- Deep Packet can identify traffic at both granular levels (application identification and traffic characterization) with state-of-the-art results compared to the other works conducted on similar dataset (Gil et al. 2016; Yamansavascilar et al. 2017).
- Deep Packet can accurately classify one of the hardest class of applications, known to be P2P (Khalife et al. 2014). This kind of applications routinely uses advanced port obfuscation techniques, embedding their information in well-known protocols’ packets and using random ports to circumvent ISPs’ controlling processes.

The rest of paper is organized as follows. In Sect. 2,we review some of the most important and recent studies on network traffic classification. In Sect. 3, we present the essential background on deep learning which is necessary to our work. Section 4 presents our proposed method, i.e., Deep Packet. The results of the proposed scheme on network application identification and traffic characterization tasks are described in Sect. 5. In Sect. 6, we provide further discussion on experimental results. Section 7 discusses future work and possible direction for further inspection. Finally, we conclude the paper in Sect. 8.

### 2 related works

In this section, we provide an overview of the most important network traffic classification methods. In particular, we can categorize these approaches into three main categories as follows: (I) port-based methods, (II) payload inspection techniques and (III) statistical and machine learning approaches. Here is a brief review of the most important and recent studies regarding each of the approaches mentioned above.

**Port-based approach** Traffic classification via port number is the oldest and the most well-known method for this task (Dainotti et al. 2012). Port-based classifiers use the information in the TCP/UDP headers of the packets to extract the port number which is assumed to be associated with a particular application. After the extraction of the port number, it is compared with the assigned IANA TCP/UDP port numbers for traffic classification. The extraction is an easy procedure, and port numbers will not be affected by encryption schemes. Because of the fast extraction process, this method is often used in firewalls and access control lists (ACL) (Qi et al. 2009). Port-based classification is known to be among the simplest and fastest method for network traffic identification. However, the pervasiveness of port obfuscation, network address translation (NAT), port forwarding, protocol embedding and random ports assignments have significantly reduced the accuracy of this approach. According to Moore and Papagiannaki (2005) and Madhukar and Williamson (2006), only 30% to 70% of the current Internet traffic can be classified using port-based classification methods. For these reasons, more complex traffic classification methods are needed to classify modern network traffic.

**Payload inspection techniques** These techniques are based on the analysis of information available in the application layer payload of packets (Khalife et al. 2014). Most of the payload inspection methods, also known as deep packet inspection (DPI), use predefined patterns like regular expressions as signatures for each protocol (e.g., see Yeganeh et al. 2012; Sen et al. 2004). The derived patterns are then used to distinguish protocols form each other. The need for updating patterns whenever a new protocol is released, and user privacy issues are among the most important drawbacks of this approach. Sherry et al. proposed a new DPI system that can inspect encrypted payload without decryption, thus solved the user privacy issue, but it can only process HTTP Secure (HTTPS) traffic (Sherry et al. 2015).

**Statistical and machine learning approach** Some of these methods, mainly known as statistical methods, have a biased assumption that the underlying traffic for each application has some statistical features which are almost unique to each application. Each statistical method uses its own functions and statistics. Crotti et al. (2007) proposed protocol fingerprints based on the probability density function (PDF) of packets inter-arrival time and normalized thresholds. They achieved up to 91% accuracy for a group of protocols such as HTTP, Post Office Protocol 3 (POP3) and Simple Mail Transfer Protocol (SMTP). In a similar work, Wang and Parish (2010) have considered PDF of the packet size. Their scheme was able to identify a broader range of protocols including file transfer protocol (FTP), Internet Message Access Protocol (IMAP), SSH, and TELNET with accuracy up to 87%.

A vast number of machine learning approaches have been published to classify traffic. Auld et al. proposed a Bayesian neural network that was trained to classify most well-known P2P protocols including Kazaa, BitTorrent, GnuTella, and achieved 99% accuracy (Auld et al. 2007). Moore et al. achieved 96% of accuracy on the same set of applications using a Naive Bayes classifier and a kernel density estimator (Moore and Zuev 2005). Artificial neural network (ANN) approaches were proposed for traffic identification (e.g., see Sun et al. 2010; Ting et al. 2010). Moreover, it was shown in Ting et al. (2010) that the ANN approach can outperform Naive Bayes methods. Two of the most important papers that have been published on “ISCX VPN-nonVPN” traffic dataset are based on machine learning methods. Gil et al. (2016) used time-related features such as the duration of the flow, flow bytes per second, forward and backward inter-arrival time, etc. to characterize the network traffic using k-nearest neighbor (k-NN) and C4.5 decision tree algorithms. They achieved approximately 92% recall, characterizing six major classes of traffic including Web browsing, email, chat, streaming, file transfer and VoIP using the C4.5 algorithm. They also achieved approximately 88% recall using the C4.5 algorithm on the same dataset which is tunneled through VPN. Yamansavascilar et al. manually selected 111 flow fea- tures described in Moore et al. (2013) and achieved 94% of accuracy for 14 class of applications using k-NN algorithm (Yamansavascilar et al. 2017). The main drawback of all these approaches is that the feature extraction and feature selection phases are essentially done with the assistance of an expert. Hence, it makes these approaches time-consuming, expensive and prone to human mistakes. Moreover, note that for the case of using k-NN classifiers, as suggested by Yamansavascilar et al. (2017), it is known that, when used for prediction, the execution time of this algorithm is a major concern.

To the best of our knowledge, prior to our work, only one study based on deep learning ideas has been reported by Wangc Wang (2015). They used stacked autoencoders (SAE) to classify some network traffic for a large family of protocols like HTTP, SMTP, etc. However, in their technical report, they did not mention the dataset they used. Moreover, the methodology of their scheme, the details of their implementation, and the proper report of their result is missing.

### 3 Background on deep neural networks

Neural networks (NNs) are computing systems made up of some simple, highly interconnected processing elements, which process information by their dynamic state response to external inputs (Caudill 1987). In practice, these networks are typically constructed from a vast number of building blocks called neuron where they are connected via some links to each other. These links are called connections, and to each of them, a weight value is associated. During the training procedure, the NN is fed with a large number of data samples. The widely used learning algorithm to train such networks (called backpropagation) adjusts the weights to achieve the desired output from the NN. The deep learning framework can be considered as a particular kind of NNs with many (hidden) layers. Nowadays, with the rapid growth of computational power and the availability of graphical processing units (GPUs), training deep NNs have become more plausible. Therefore, the researchers from different scientific fields consider using deep learning framework in their respective area of research, e.g., see Hinton et al. (2012), Lotfollahi et al. (2018) and Socher et al. (2013). In the following, we will briefly review two of the most important deep neural networks that have been used in our proposed scheme for network traffic classification, namely autoencoders and convolutional neural networks.

#### 3.1 Autoencoder

An autoencoder NN is an unsupervised learning framework that aims to reconstruct the input at the output while minimizing the reconstruction error (i.e., according to some criteria). Consider a training set {x1, x2,...,xn } where for each training data we have xi ∈ Rn. The autoencoder’s objective is defined to be yi = xi for i ∈{1, 2,...,n}, i.e., the output of the network will be equal to its input. Considering this objective function, the autoencoder tries to learn a compressed representation of the dataset, i.e., it approximately learns the identity function FW ,b(x )  x, where W and b are the whole network weights and biases vectors. General form of an autoencoder’s loss function is shown in (1), as follows

$$function$$

Figure 1 shows a typical autoencoder with n inputs and outputs. The autoencoder is mainly used as an unsupervised technique for automatic feature extraction. More precisely, the output of the encoder part is considered as a high-level set of discriminative features for the classification task.

In practice, to obtain a better performance, a more complex architecture and training procedure, called stacked autoen-coder (SAE), is proposed (Vincent et al. 2008). This scheme suggests to stack up several autoencoders in a manner that output of each one is the input of the successive layer which itself is an autoencoder. The training procedure of a stacked autoencoder is done in a greedy layer-wise fashion (Bengio et al. 2007). First, this method trains each layer of the network while freezing the weights of other layers. After training all the layers, to have more accurate results, fine-tuning is applied to the whole NN. At the fine-tuning phase, the backpropagation algorithm is used to adjust all layers’ weights. Moreover, for the classification task, an extra softmax layer can be applied to the final layer. Figure 2 depicts the training procedure of a stacked autoencoder.

#### 3.2 Convolutional neural network

The convolutional neural networks (CNN) are another types of deep learning models in which feature extraction from the input data is done using layers comprised of convolutional operations (i.e., convolutional filters). The construction of convolutional networks is inspired by the visual structure of living organisms (Hubel and Wiesel 1968). Basic building block underneath a CNN is a convolutional layer described as follows. Consider a convolutional layer with N × N square neuron layer as input and a filter ω of size m × m. The output of this layer zl is of size (N − m + 1) × (N − m + 1) and is computed as follows

$$function$$

As it is demonstrated in (2), a nonlinear function f such as rectified linear unit (ReLU) is applied to the convolution output to learn more complex features from the data. In some applications, a pooling layer (e.g., max pooling) is also applied. The main motivation of employing a pooling layer is to aggregate multiple low-level features in a neighborhood to obtain local invariance. Moreover, by reducing the output size, it helps to reduce the computation cost of the network in train and test phase.

CNNs have been successfully applied to different fields including natural language processing (dos Santos and Gatti 2014), computational biology (Alipanahi et al. 2015), and machine vision (Simonyan and Zisserman 2014). One of the most interesting applications of CNNs is in face recognition (Lee et al. 2009), where consecutive convolutional layers are used to extract features from each image. It is observed that the extracted features in shallower layers are simple concepts like edges and curves. On the contrary, features in deeper layers of networks are more abstract than the ones in shallower layers (Yosinski et al. 2015). However, it is worth mentioning that visualizing the extracted features in the middle layers of a network does not always lead to meaningful concepts like what has been observed in the face recognition task. For example in one-dimensional CNN (1D-CNN) which we use to classify network traffic, the feature vectors extracted in shallow layers are just some real numbers which make no sense at all for a human observer.

We believe 1D-CNNs are an ideal choice for the network traffic classification task. This is true since 1D-CNNs can capture spatial dependencies between adjacent bytes in network packets that leads to find discriminative patterns for every class of protocols/applications, and consequently, an accurate classification of the traffic. Our classification results confirm this claim and prove that CNNs performs very well in feature extraction of network traffic data.

### 4 Methodology

In this work, we develop a framework, called Deep Packet, that comprises two deep learning methods, namely convolutional NN and stacked autoencoder NN, for both “application identification” and “traffic characterization” tasks. Before training the NNs, we have to prepare the network traffic data so that it can be fed into NNs properly. To this end, we perform a pre-processing phase on the dataset. Figure 3 demonstrates the general structure of Deep Packet. At the test phase, a pre-trained neural network corresponding to the type of classification, application identification or traffic characterization, is used to predict the class of traffic the packet belongs to. The dataset, implementation and design details of the pre-processing phase and the architecture of proposed NNs will be explained in the following.

#### 4.1 Dataset

For this work, we use “ISCX VPN-nonVPN” traffic dataset, that consists of captured traffic of different applications in pcap format files (Gil et al. 2016). In this dataset, the captured packets are separated into different pcap files labeled according to the application produced the packets (e.g., Skype, and Hangouts, etc.) and the particular activity the application was engaged during the capture session (e.g., voice call, chat, file transfer, or video call). For more details on the captured traffic and the traffic generation process, refer to Gil et al. (2016).

The dataset also contains packets captured over Virtual Private Network (VPN) sessions. A VPN is a private overlay network among distributed sites which operates by tunneling traffic over public communication networks (e.g., the Internet). Tunneling IP packets, guaranteeing secure remote access to servers and services, is the most prominent aspect of VPNs (Chowdhury and Boutaba 2010). Similar to regular (non-VPN) traffic, VPN traffic is captured for different applications, such as Skype, while performing different activities, like voice call, video call, and chat.

Furthermore, this dataset contains captured traffic of Tor software. This traffic is presumably generated while using Tor browser, and it has labels such as Twitter, Google, Facebook, etc. Tor is a free, open source software developed for anonymous communications. Tor forwards users’ traffic through its own free, worldwide, overlay network which consists of volunteer-operated servers. Tor was proposed to protect users against Internet surveillance known as “traffic analysis.” To create a private network pathway, Tor builds a circuit of encrypted connections through relays on the network in a way that no individual relay ever knows the complete path that a data packet has taken (Dingledine et al. 2004). Finally, Tor uses complex port obfuscation algorithm to improve privacy and anonymity.

#### 4.2 Pre-processing

The “ISCX VPN-nonVPN” dataset is captured at the data-link layer. Thus, it includes the Ethernet header. The data-link header contains information regarding the physical link, such as Media Access Control (MAC) address, which is essential for forwarding the frames in the network, but it is uninformative for either the application identification or traffic characterization tasks. Hence, in the pre-processing phase, the Ethernet header is removed first. Transport layer segments, specifically Transmission Control Protocol (TCP) or User Datagram Protocol (UDP), vary in header length. The former typically bears a header of 20 bytes length, while the latter has an 8 bytes header. To make the transport layer segments uniform, we inject zeros to the end of UDP segment’s headers to make them equal length with TCP headers. The packets are then transformed from bits to bytes which helps to reduce the input size of the NNs.

Since the dataset is captured in a real-world emulation, it contains some irrelevant packets which are not of our interest and should be discarded. In particular, the dataset includes some TCP segments with either SYN, ACK, or FIN flags set to one and containing no payload. These segments are needed for three-way handshaking procedure while establishing a connection or finishing one, but they carry no information regarding the application generated them, thus can be safely discarded. Furthermore, there are some Domain Name Service (DNS) segments in the dataset. These segments are used for hostname resolution, namely translating URLs to IP addresses. These segments are not relevant to either application identification or traffic characterization, hence can be omitted from the dataset.
Figure 4 illustrates the histogram (empirical distribution) of packet length for the dataset. As the histogram shows, packet length varies a lot through the dataset, while employing NNs necessitates using a fixed-size input. Hence, truncation at a fixed length or zero-padding is required inevitably. To find the fixed length for truncation, we inspected the packets length’s statistics. Our investigation revealed that approximately 96% of packets have a payload length of less than 1480 bytes. This observation is not far from our expectation, as most of the computer networks are constrained by Maximum Transmission Unit (MTU) size of 1500 bytes. Hence, we keep the IP header and the first 1480 bytes of each IP packet which results in a 1500 bytes vector as the input for our proposed NNs. Packets with IP payload less than 1480 bytes are zero-padded at the end. To obtain a better perfor- mance, all the packet bytes are divided by 255, the maximum value for a byte, so that all the input values are in the range [0, 1].

Furthermore, since there is the possibility that the NN attempts to learn classifying the packets using their IP addresses, as the dataset is captured using a limited number of hosts and servers, we decided to prevent this over-fitting by masking the IP addresses in the IP header. In this matter, we assure that the NN is not using irrelevant features to perform classification. All of the pre-processing steps mentioned above take place when the user loads a pcap file into Deep Packet toolkit.

##### 4.2.1 Labeling dataset

As mentioned before in Sect. 4.1, the dataset’s pcap files are labeled according to the applications and activities they were engaged in. However, for application identification and traffic characterization tasks, we need to redefine the labels, concerning each task. For application identification, all pcap files labeled as a particular application which were collected during a non-VPN session are aggregated into a single file. This leads to 17 distinct labels shown in Table 1a. Also for traffic characterization, we aggregated the captured traffic of different applications involved in the same activity, taking into account the VPN or non-VPN condition, into a single pcap file. This leads to a 12-class dataset, as shown in Table 1b. By observing Table 1, one would instantly notice that the dataset is significantly imbalanced and the number of samples varies remarkably among different classes. It is known that such an imbalance in the training data leads to a reduced classification performance. Sampling is a simple yet powerful technique to overcome this problem (Longadge and Dongre 2013). Hence, to train the proposed NNs, using the undersampling method, we randomly remove the major classes’ samples (classes having more samples) until the classes are relatively balanced.

#### 4.3 Architectures 

In the following, we explain our two proposed architectures used in the Deep Packet toolkit.

The proposed SAE architecture consists of five fully connected layers, stacked on top of each other which made up of 400, 300, 200, 100 and 50 neurons, respectively. To prevent the over-fitting problem, after each layer the dropout technique with 0.05 dropout rate is employed. In this technique, during the training phase, some of the neurons are set to zero randomly. Hence, at each iteration, there is a random set of active neurons. For the application identification and traffic characterization tasks, at the final layer of the proposed SAE, a softmax classifier with 17 and 12 neurons is added, respectively.

A minimal illustration of the second proposed scheme, based on one-dimensional (1D) CNN, is depicted in Fig. 5. We used a grid search on a subspace of the hyper-parameters space to select the ones which results in the best performance. This procedure is discussed in detail in Sect. 5. Our final proposed model consists of two consecutive convolutional layers, followed by a pooling layer. Then, the two-dimensional tensor is squashed into a one-dimensional vector and fed into a three-layered network of fully connected neurons which also employ dropout technique to avoid over-fitting. Finally, a softmax classifier is applied for the classification task, similar to the SAE architecture. The best values found for the hyper-parameters are shown in Table 2. The detailed architecture of all the proposed models for application identification and traffic characterization tasks can be found in “Appendix A”.

### 5 Experimental results

To implement our proposed NNs, we have used Keras library (Chollet et al 2017), with Tensorflow (Abadi et al. 2015)as its backend. Each of the proposed models was trained and evaluated against the independent test set that was extracted from the dataset. We randomly split the dataset into three separate sets. The first one which includes 64% of samples is used for training and adjusting weights and biases. The second part containing 16% of samples is used for validation during the training phase, and finally the third set made up of 20% of data points is used for testing the model. Additionally, to avoid the over-fitting problem, we have used early stopping technique (Prechelt 1998). This technique stops the training procedure, once the value of loss function on the validation set remains almost unchanged for several epochs, and thus prevents the network to over-fit on the training data. To speed up the learning phase, we also used Batch Normalization technique in our models (Ioffe and Szegedy 2015).

For training SAE, first each layer was trained in a greedy layer-wise fashion using Adam optimizer (Kingma and Ba 2014) and mean squared error as the loss function for 200 epochs, as described in Sect. 3.1. Next, in the fine-tuning phase, the whole network was trained for another 200 epochs using the categorical cross entropy loss function. Also, for implementing the proposed one-dimensional CNN, the categorical cross entropy and Adam were used as loss function and optimizer, respectively, and in this case, the network was trained for 300 epochs. Finally, it is worth mentioning that in both NNs, all layers employ Rectified Linear Unit (ReLU) as the activation function, except for the final softmax classifier layer.

To evaluate the performance of Deep Packet, we have used Recall (Rc), Precision (Pr) and F1 Score (i.e., F1) metrics. The above metrics are described mathematically as follows

$$function$$

where TP, FP and FN stand for true positive, false positive and false negative, respectively.

As mentioned in Sect. 4, we used grid search hyper-parameters tuning scheme to find the best 1D-CNN structure in our work. Due to our computation hardware limitations, we only searched a restricted subspace of hyper-parameters 5 Experimental results to find the ones which maximize the weighted average F1 score on the test set for each task. To be more specific, we changed filter size, the number of filters and stride for both convolutional layers. In total, 116 models with their weighted average F1 score for both application identification and traffic characterization tasks were evaluated. The result for all trained models can be seen in Fig. 6. We believe one cannot select an optimal model for traffic classification tasks since the definition of “optimal model” is not well defined and there exists a trade-off between the model accuracy and its complexity (i.e., training and test speed). In Fig. 6, the color of each point is associated with the model’s trainable parameters; the darker the color, the higher the number of trainable parameters.

As seen in Fig. 6, increasing the complexity of the neural network does not necessarily result in a better performance. Many reasons can cause this phenomenon which among them one can mention to the vanishing gradient and over-fitting problems. A complex model is more likely to face the vanishing gradient problem which leads to under-fitting in the training phase. On the other hand, if a learning model becomes more complex while the size of training data remains the same, the over-fitting problem can be occurred. Both of these problems lead to a poor performance of NNs in the evaluation phase.

Table 3 shows the achieved performance of both SAE and 1D-CNN for the application identification task on the test set. The weighted average F1 score of 0.98 and 0.95 for 1D-CNN and SAE, respectively, shows that our networks have entirely extracted and learned the discriminating features from the training set and can successfully distinguish each application. For the traffic characterization task, our proposed CNN and SAE have achieved F1 score of 0.93 and 0.92, respectively, implying that both networks are capable of accurately classify packets. Table 4 summaries the achieved performance of the proposed methods on the test set.
 
#### 5.1 Comparison

In the following, we compare the results of Deep Packet with previous results using the “ISCX VPN-nonVPN” dataset. Moreover, the Deep Packet is compared against some of the other machine learning methods in Sect. 5.1.2.

##### 5.1.1 Comparison with previous results

As mentioned in Sect. 2, authors in Gil et al. (2016)triedto characterize network traffic using time-related features hand-crafted from traffic flows such as the duration of the flow and flow bytes per second. Yamansavascilar et al. also used such time-related features to identify the end-user application (Yamansavascilar et al. 2017). Both of these studies evaluated their models on the “ISCX VPN-nonVPN traffic dataset,” and their best results can be found in Table 5. The results suggest that Deep Packet has outperformed other proposed approaches mentioned above, in both application identification and traffic characterization tasks.

We would like to emphasize that the above-mentioned work have used handcrafted features based on the network traffic flow. On the other hand, Deep Packet considers the network traffic in the packet level and can classify each packet of network traffic flow which is a harder task, since there is more information in a flow compared to a single packet. This feature allows Deep Packet to be more applicable in real-world situations.

Finally, it worth mentioning that independently and parallel to our work (Lotfollahi et al. 2017), Wang et al. proposed a similar approach to Deep Packet for traffic characterization on “ISCX VPN-nonVPN” traffic dataset (Wang et al. 2017). Their best-reported result achieves 100% precision on the traffic characterization task. However, we believe that their result is seriously questionable. The proving reason for our allegation is that their best result has been obtained by using packets containing all the headers from every five layers of the Internet protocol stack. However, based on our experiments and also a direct inquiry from the dataset providers (Gil et al. 2016), in “ISCX VPN-nonVPN” traffic dataset, the source and destination IP addresses (that are appeared in the header of network layer) are unique for each application. Therefore, their model presumably just uses this feature to classify the traffic (in that case a much simpler classifier would be sufficient to handle the classification task). As mentioned before, to avoid this phenomenon, we mask IP address fields in the pre-processing phase before feeding the packets into our NNs for training or testing.

##### 5.1.2 Comparison with previous methods

In this section, we compare Deep Packet with four machine learning algorithms. The comparison was performed by feeding pre-possessed packets similar to what we feed to Deep packet. We used scikit-learn (Pedregosa et al. 2011)implementation of the decision tree with depth two, random forests with depth four, logistic regression (with c = 0.1) and naive Bayes with default parameters. Table 6 indicates our method outperforms four alternative algorithms in application identification task for the test data. Similarly, Table 7 illustrates Deep Packet performs better in traffic characterization task.

These comparisons confirm the power of deep neural network for the network traffic classification where a huge amount of data have to be analyzed.

### 6 Discussion

Evaluating the SAE on the test set for the application identification and the traffic characterization tasks result in row-normalized confusion matrices shown in Fig. 7.The rows of the confusion matrices correspond to the actual class of the samples, and the columns present the predicted label; thus, the matrices are row-normalized. The dark color of the elements on the main diagonal suggests that SAE can classify each application with minor confusion.

By carefully observing the confusion matrices in Fig. 7, one would notice some interesting confusion between different classes (e.g., ICQ and AIM). Hierarchical clustering further demonstrates the similarities captured by Deep Packet. Clustering on row-normalized confusion matrices for application identification with SAE (Fig. 7a), using Euclidean distance as the distance metric and Ward.D as the agglomeration method uncovers similarities among applications regarding their propensities to be assigned to the 17 application classes. As illustrated in Fig. 8a, application groupings revealed by Deep Packet generally agree with the applications’ similarities in the real world. Hierarchical clustering divided the applications into 7 groups. Interestingly, these groups are to some extent similar to groups in the traffic characterization task. One would notice that Vimeo, Netflix, YouTube and Spotify which are bundled together are all streaming applications. There is also a cluster including ICQ, AIM, and Gmail. AIM and ICQ are used for online chatting, and Gmail in addition to email services offers a service for online chatting. Another interesting observation is that Skype, Facebook, and Hangouts are all grouped in a cluster together. Though these applications do not seem much relevant, this grouping can be justified. The dataset contains traffic for these applications in three forms: voice call, video call, and chat. Thus, the network has found these applications similar regarding their usage. FTPS (File Transfer Protocol over SSL) and SFTP (File Transfer Protocol over SSH) which are both used for transferring files between two remote systems securely are clustered together as well. Interestingly, SCP (Secure Copy) has formed its cluster although it is also used for remote file transfer. SCP uses SSH protocol for transferring file, while SFTP and FTPS use FTP. Presumably, our network has learned this subtle difference and separated them. Tor and Torrent have their clusters which are sensible due to their apparent differences with other applications. This clustering is not flawless. Clustering Skype, Facebook, and Hangouts along with Email and VoipBuster are not correct. VoipBuster is an application which offers voice communications over Internet infrastructure. Thus, applications in this cluster do not seem much similar regarding their usage, and this grouping is not precise.

The same procedure was performed on the confusion matrices of traffic characterization as illustrated in Fig. 8b. Interestingly, groupings separate the traffic into VPN and non-VPN clusters. All the VPN traffics are bundled together in one cluster, while all of non-VPNs are grouped together.

As mentioned in Sect. 2, many of the applications employ encryption to maintain clients’ privacy. As a result, the majority of “ISCX VPN-nonVPN” dataset traffics are also encrypted. One might wonder how it is possible for Deep Packet to classify such encrypted traffics. Unlike DPI methods, Deep Packet does not inspect the packets for keywords. In contrast, it attempts to learn features in traffic generated by each application. Consequently, it does not need to decrypt the packets to classify them.

An ideal encryption scheme causes the output message to bear the maximum possible entropy (Cover and Thomas 2006). In other words, it produces patternless data that theoretically cannot be distinguished from one another. However, due to the fact that all practical encryption schemes use pseudo-random generators, this hypothesis is not valid in practice. Moreover, each application employs different (non-ideal) ciphering scheme for data encryption. These schemes utilize different pseudo-random generator algorithms which leads to distinguishable patterns. Such variations in the pattern can be used to separate applications from one another. Deep Packet attempts to extract those discriminative patterns and learns them. Hence, it can classify encrypted traffic accurately.

It is noticeable from Table 3 that Tor traffic is also successfully classified. To further investigate this kind of traffic, we conducted another experiment in which we trained and tested Deep Packet with a dataset containing only Tor traffic. To achieve the best possible result, we performed a grid search on the hyper-parameters of the NN, as discussed before. The detailed results can be found in Table 8, which shows that Deep Packet was unable to classify the underlying Tor’s traffic accurately. This phenomenon is not far from what we expected. Tor encrypts its traffic, before transmission. As mentioned earlier, Deep Packet presumably learns different pseudo-random patterns used in various encryption schemes used by applications. At this experiment, traffic was tunneled through Tor. Hence, they all experience the same encryption scheme. Consequently, our neural network was not able to separate them apart well.

### 7 Future work

The reasons why deep neural networks perform so well in practice are yet to be understood. In addition, there is no rigorous theoretical framework to design and analyze such networks. If there is some progress in these matters, it will have direct impact on proposing better deep neural network structures specialized for network traffic classification. Along the same line, one of the other important future direction would be investigating the interpretability (Du et al. 2018; Montavon et al. 2018; Samek et al. 2018) of our proposed model. This will include analyzing the features that the model has learned and the process of learning them.

Another important direction to be studied would be the robustness analysis of proposed schemes against noisy and maliciously generated inputs using adversarial attack algorithms (Yuan et al. 2017). Adversarial attacks on machine learning methods have been widely studied in some other fields (e.g., Akhtar and Mian 2018; Huang et al. 2017; Carlini and Wagner 2018) but not in network traffic classification.

Designing multi-level classification algorithms is also an interesting possible direction for future research. This means that the system should be able to detect whether a traffic is from one of the known previous classes or a new “unknown” class. If the packet is labeled as unknown, then it will be added to a database of unknown classes. Further, by receiving more unknown packets, one can use an unsupervised clustering algorithm to label them as discrete classes. Next, human experts will be able to map these unknown classes to well-known real-world applications. Thus, re-training the first level classifier would become possible with these new labeled classes. Re-training can be done with an online learning algorithm or using previously learned weights of the neural network as initialization for the newer network.

Finally, implementing the proposed schemes to be able to handle the real-world high-speed network traffic will be an important real challenge. This can be done for example by taking advantage of hardware implementation (e.g., see Vanhoucke et al. 2011; Zhang et al. 2015) and applying neural network simplification techniques (e.g., see Hubara et al. 2017; Lin et al. 2016).

### 8 Conclusion

In this paper, we presented Deep Packet, a framework that automatically extracts features from computer networks traffic using deep learning algorithms to classify traffic. To the best of our knowledge, Deep Packet is the first traffic classification system using deep learning algorithms, namely SAE and 1D-CNN that can handle both application identification and traffic characterization tasks. Our results showed that Deep Packet outperforms all of the similar works on the “ISCX VPN-nonVPN” traffic dataset, in both application identification and traffic characterization tasks, to the date. Moreover, with state-of-the-art results achieved by Deep Packet, we envisage that Deep Packet is the first step toward a general trend of using deep learning algorithms in traffic classification and more generally network analysis tasks. Furthermore, Deep Packet can be modified to handle more complex tasks like multi-channel (e.g., distinguishing between different types of Skype traffic including chat, voice call, and video call) classification, accurate classification of Tor’s traffic, etc. Finally, the automatic feature extraction procedure from network traffic can save the cost of employing experts to identify and extract handcrafted features from the traffic which eventually leads to more accurate traffic classification.


## 深度包：一种使用深度学习进行加密流量分类的新方法

### 作者

Mohammad Lotfollahi1 · Mahdi Jafari Siavoshani1 · Ramin Shirali Hossein Zade1 · Mohammdsadegh Saberian1

### 摘要

随着互联网和在线应用的快速增长，网络流量分类变得越来越重要。在这个主题上已经有了许多研究，产生了许多不同的方法。这些方法中的大多数使用专家提取的预定义特征来对网络流量进行分类。相比之下，在本研究中，我们提出了一种基于深度学习的方法，将特征提取和分类阶段集成到一个系统中。我们提出的方案，称为“深度数据包”，既可以处理将网络流量分为主要类别（例如，FTP 和 P2P）的流量分类任务，也可以处理需要识别最终用户应用程序（例如，BitTorrent 和 Skype）的应用程序识别任务。与当前大多数方法相反，深度包方法可以识别加密流量，还可以区分 VPN 和非 VPN 网络流量。深度包框架采用两种深度神经网络结构，即堆栈自编码器和卷积神经网络来对网络流量进行分类。我们的实验表明，当深度包方法使用卷积神经网络作为其分类模型时，它在应用识别任务中实现了 0.98 的召回率和在流量分类任务中实现了 0.94 的召回率，从而获得了最佳结果。据我们所知，深度包在 UNB ISCX VPN-nonVPN 数据集上优于所有已经提出的分类方法。

### 关键字

网络流量分类 · 应用识别 · 流量分类 · 深度学习 · 卷积神经网络 · 堆栈自编码器 · 深度包

### 1 简介

流量分类是现代通信网络中的一项重要任务（Bagui et al. 2017）。由于高吞吐量流量需求的快速增长，为了正确管理网络资源，识别利用网络资源的不同类型的应用程序至关重要。因此，准确的流量分类已成为诸如提供适当的服务质量、异常检测、定价等高级网络管理任务的先决条件之一。流量分类已引起与网络管理相关的学术界和工业界的广泛关注。（例如，参见 Dainotti 等人 2012；Finsterbusch 等人 2014；Velan 等人 2015 及其中的参考文献。）

作为网络流量分类重要性的一个例子，可以考虑当今网络访问链路的非对称架构，该架构是基于客户端下载多于上传的假设而设计的。然而，对称需求应用程序（例如 P2P 应用程序、IP 语音和视频通话）的普遍存在已经改变了客户的需求，使其偏离了前面提到的假设。因此，为了给客户提供满意的体验，需要了解到应用级别的情况来为这些应用分配足够的资源。

互联网上新应用的出现以及各种组件之间的交互极大地增加了该网络的复杂性和多样性，这使得流量分类本身成为一个难题。在下文中，我们将详细讨论网络流量分类的一些最关键的挑战。

首先，对用户隐私保护和数据加密的需求不断增长，极大地增加了当今互联网的加密流量（Velan et al. 2015）。加密过程将原始数据转换为类似伪随机的格式，目的是使其难以解密。结果，它导致加密数据几乎不包含任何可以识别网络流量的判别模式。因此，加密流量的准确分类已成为现代网络中的真正挑战（Dainotti et al. 2012）。

还值得一提的是，许多已经提出的网络流量分类方法，例如有效载荷检查以及基于机器学习和基于统计的方法，都需要专家提取模式或特征。这个过程容易出错、耗时且成本高。

最后，许多互联网服务提供商因为高带宽消耗和版权问题而阻止 P2P 文件共享应用程序(Lv et al. 2014)。因此，为了规避这个问题，这些应用程序使用协议嵌入和混淆技术来绕过流量控制系统（Alshammari 和 Zincir-Heywood 2011）。识别此类应用程序是网络流量分类中最具挑战性的任务之一。

关于网络流量分类主题已有大量研究，例如 Kohout 和 Pevný (2018)、Perera 等。 (2017)、Giletal.(2016) 和 Moore 和 Papagiannaki (2005)。但是，他们中的大多数都专注于对协议族进行分类，也称为流量分类（例如，流式传输、聊天、P2P 等），而不是识别单个应用程序，这称为应用程序识别（例如，Spotify、Hangouts） 、BitTorrent 等）（Khalife 等人，2014 年）。相比之下，这项工作提出了一种方法，即深度包，它基于机器学习社区最近开发的思想，即深度学习 (Bengio 2009; LeCun et al. 2015)，用于分类和识别网络流量。我们提出的方法的优点使其优于其他分类方案，如下所述：

- 在深度包方法中，不需要专家来提取与网络流量相关的特征。鉴于这种方法，查找和提取区别特征的繁琐步骤已被省略。
- 与在类似数据集上进行的其他工作相比，深度包方法可以在两个粒度级别（应用程序识别和流量分类）上识别流量，并具有最先进的结果（Gil et al. 2016; Yamansavascilar et al. 2017）。
- 深度包可以准确分类最难的一类应用程序，即 P2P (Khalife et al. 2014)。这类应用程序通常使用先进的端口混淆技术，将其信息嵌入知名协议的数据包中，并使用随机端口绕过互联网服务提供商的控制过程。

其余的论文组织如下。在第二部分我们回顾了一些关于网络流量分类的最重要和最近的研究。在第三部分我们介绍了我们工作所必需的深度学习的基本背景。在第四部分介绍了我们提出的方法，即深度包。我们所提出的网络应用程序识别和流量分类任务方案的结果在第五部分中进行了描述。而在第六部分我们对实验结果进行了进一步的讨论。在第七部分讨论了未来的工作和进一步查看的可能方向。最后，我们在第八部分总结了整片论文。

### 2 相关工作

在本节中，我们概述了最重要的网络流量分类方法。特别是，我们可以将这些方法分为以下三大类：（I）基于端口的方法，（II）有效载荷检测技术和（III）统计和机器学习方法。以下是对上述每种方法的最重要和最新研究的简要回顾。

**基于端口的方法** 通过端口号进行流量分类是这项任务最古老和最著名的方法（Dainotti et al. 2012）。基于端口的分类器使用数据包的 TCP/UDP 标头中的信息来提取假定与特定应用程序相关联的端口号。提取端口号后，与分配的 IANA TCP/UDP 端口号进行比较，进行流量分类。提取过程简单，端口号不受加密方案的影响。由于提取过程快速，这种方法经常用于防火墙和访问控制列表 (ACL) (Qi et al. 2009)。众所周知，基于端口的分类是网络流量识别最简单、最快的方法之一。然而，端口混淆、网络地址转换、端口转发、协议嵌入和随机端口分配的普遍性大大降低了这种方法的准确性。根据 Moore 和 Papagiannaki (2005) 以及 Madhukar 和 Williamson (2006) 的说法，目前只有 30% 到 70% 的互联网流量可以使用基于端口的分类方法进行分类。由于这些原因，需要更复杂的流量分类方法来对现代网络流量进行分类。

**有效载荷检测技术** 这些技术基于对数据包应用层有效载荷中可用信息的分析（Khalife et al. 2014）。大多数有效负载检测方法，也称为深度数据包检测，使用预定义的模式（如正则表达式）作为每个协议的签名（例如，参见 Yeganeh 等人 2012；Sen 等人 2004）。然后使用派生的模式来区分协议彼此。每当发布新协议时都需要更新模式，并且用户隐私问题是这种方法最重要的缺点之一。Sherry等人提出了一种新的深度数据包检测系统，可以在不解密的情况下检查加密的有效负载，从而解决了用户隐私问题，但它只能处理 HTTP 安全流量（Sherry et al. 2015）。

**统计和机器学习方法** 其中一些方法，主要称为统计方法，有一个有偏见的假设，即每个应用程序的基础流量具有一些几乎每个应用程序独有的统计特征。每种统计方法都使用自己的功能和统计数据。Crotti等人(2007)提出了基于数据包到达时间和归一化阈值的概率密度函数的协议指纹。他们对 HTTP、POP3和SMTP等一组协议的准确率高达 91%。在类似的工作中，Wang 和 Parish (2010) 考虑了数据包大小的概率密度函数。他们的方案能够识别更广泛的协议，包括文件传输协议 (FTP)、互联网消息访问协议 (IMAP)、SSH 和 TELNET，准确率高达 87%。

已经存在了大量机器学习方法来对流量进行分类。Auld等人提出了一个贝叶斯神经网络，该网络经过训练可以对包括 Kazaa、BitTorrent、GnuTella 在内的大多数知名 P2P 协议进行分类，并达到 99% 的准确率。Moore等人使用朴素贝叶斯分类器和核密度估计器在同一组应用程序上实现了 96% 的准确度（Moore 和 Zuev 2005）。人工神经网络方法被提出用于交通识别（例如，参见 Sun 等人 2010；Ting 等人 2010）。此外，它在 Ting 等人的工作中有所体现(2010)。人工神经网络方法可以胜过朴素贝叶斯方法。在“ISCX VPN-nonVPN”流量数据集上发表的两篇最重要的论文都是基于机器学习方法的。Gil等人(2016) 使用与时间相关的特征，例如流的持续时间、每秒流字节数、前向和后向到达间隔时间等，使用 k-最近邻和 C4.5决策树算法来分类网络流量。他们实现了大约 92% 的召回率，使用 C4.5 算法分类了六大类流量，包括网站浏览、电子邮件、聊天、流媒体、文件传输和 IP 语音。他们还在通过 VPN 隧道传输的同一数据集上使用 C4.5 算法实现了大约 88% 的召回率。Yamansavascilar等人手动选择 Moore 等人描述的 111 个流动特征并使用 k-最近邻算法在 14 类应用中实现了 94% 的准确率（Yamansavascilar 等人，2017 年）。所有这些方法的主要缺点是特征提取和特征选择阶段基本上是在专家的帮助下完成的。因此，它使这些方法耗时、昂贵且容易出现人为错误。此外，请注意，如果按照 Yamansavascilar 等人的建议，对于使用 k-最近邻分类器的情况，跟我们了解的一样，当用于预测时，该算法的执行时间是一个主要问题。

据我们所知，在我们工作之前，只有一项 Wang 等人在2015年发表的基于深度学习思想的研究。他们使用堆栈式自动编码器 (SAE) 对 HTTP、SMTP 等一大类协议的一些网络流量进行分类。但是，在他们的技术报告中，没有提到他们使用的数据集。此外，他们的方案的方法、实施的细节以及对结果的正确报告都缺失了。

### 3 深度神经网络背景

神经网络 (NN) 是由一些简单的、高度互连的处理元素组成的计算系统，它们通过对外部输入的动态响应来处理信息 (Caudill 1987)。在实践中，这些网络通常由大量称为神经元的构建块构成，它们通过一些链接相互连接。这些链接称为连接，每个链接都关联一个权重值。在训练过程中，NN 被输入了大量的数据样本。用于训练此类网络的广泛使用的学习算法（称为反向传播）会调整权重以从 NN 获得所需的输出。深度学习框架可以被视为具有许多（隐藏）层的特定类型的神经网络。如今，随着计算能力的快速增长和图形处理单元 (GPU) 的可用性，训练深度神经网络变得更加合理。因此，来自不同科学领域的研究人员考虑在各自的研究领域使用深度学习框架，例如，参见 Hinton 等人。 (2012), Lotfollahi 等人。 （2018 年）和 Socher 等人。 （2013）。在下文中，我们将简要回顾在我们提出的网络流量分类方案中使用的两个最重要的深度神经网络，即自动编码器和卷积神经网络。

#### 3.1 自动编码器

自动编码器 NN 是一种无监督学习框架，旨在在输出端重构输入，同时最小化重构误差（即，根据某些标准）。考虑一个训练集 {x1, x2,...,xn }，其中对于每个训练数据，我们有 xi ∈ Rn。自动编码器的目标定义为 yi = xi 对于 i ∈{1, 2,...,n}，即网络的输出将等于其输入。考虑到这个目标函数，自动编码器尝试学习数据集的压缩表示，即，它近似地学习恒等函数 FW ,b(x ) x，其中 W 和 b 是整个网络的权重和偏差向量。自编码器损失函数的一般形式如（1）所示，如下

$$功能$$

图 1 显示了具有 n 个输入和输出的典型自动编码器。自动编码器主要用作自动特征提取的无监督技术。更准确地说，编码器部分的输出被认为是分类任务的高级判别特征集。

在实践中，为了获得更好的性能，提出了一种更复杂的架构和训练过程，称为堆叠自动编码器 (SAE) (Vincent et al. 2008)。该方案建议以这样的方式堆叠多个自动编码器，即每个自动编码器的输出都是连续层的输入，而连续层本身就是一个自动编码器。堆叠式自动编码器的训练过程以贪婪的逐层方式完成（Bengio et al. 2007）。首先，这种方法训练网络的每一层，同时冻结其他层的权重。在对所有层进行训练后，为了获得更准确的结果，对整个 NN 进行微调。在微调阶段，使用反向传播算法调整所有层的权重。此外，对于分类任务，可以将额外的 softmax 层应用于最后一层。图 2 描述了堆叠自动编码器的训练过程。

#### 3.2 卷积神经网络

卷积神经网络 (CNN) 是另一种类型的深度学习模型，其中使用由卷积运算（即卷积滤波器）组成的层从输入数据中提取特征。卷积网络的构建受到生物体视觉结构的启发（Hubel and Wiesel 1968）。 CNN 下的基本构建块是如下所述的卷积层。考虑一个以 N × N 方形神经元层为输入的卷积层和一个大小为 m × m 的滤波器 ω。该层 zl 的输出大小为 (N - m + 1) × (N - m + 1)，计算如下

$$function$$

如 (2) 所示，将非线性函数 f（例如整流线性单元 (ReLU)）应用于卷积输出，以从数据中学习更复杂的特征。在某些应用程序中，还应用了池化层（例如，最大池化）。使用池化层的主要动机是聚合邻域中的多个低级特征以获得局部不变性。此外，通过减小输出大小，有助于降低网络在训练和测试阶段的计算成本。

CNN 已成功应用于不同领域，包括自然语言处理 (dos Santos and Gatti 2014)、计算生物学 (Alipanahi et al. 2015) 和机器视觉 (Simonyan and Zisserman 2014)。 CNN 最有趣的应用之一是人脸识别（Lee et al. 2009），其中连续的卷积层用于从每个图像中提取特征。可以观察到，在较浅层中提取的特征是简单的概念，如边缘和曲线。相反，网络深层的特征比浅层的特征更抽象（Yosinski et al. 2015）。然而，值得一提的是，在网络的中间层可视化提取的特征并不总是会产生有意义的概念，比如在人脸识别任务中观察到的概念。例如，在我们用于对网络流量进行分类的一维 CNN (1D-CNN) 中，在浅层中提取的特征向量只是一些实数，对于人类观察者来说根本没有意义。

我们相信 1D-CNN 是网络流量分类任务的理想选择。这是真的，因为 1D-CNN 可以捕获网络数据包中相邻字节之间的空间依赖关系，从而为每一类协议/应用程序找到判别模式，从而对流量进行准确分类。我们的分类结果证实了这一说法，并证明 CNN 在网络流量数据的特征提取方面表现非常出色。

### 4 方法论

在这项工作中，我们开发了一个名为深度包的框架，它包含两种深度学习方法，即卷积神经网络和堆栈自编码器神经网络，用于“应用程序识别”和“流量分类”任务。在训练神经网络之前，我们必须准备好网络流量数据，以便可以正确地将其输入神经网络。为此，我们对数据集执行预处理阶段。图 3 展示了深度包的一般结构。在测试阶段，使用与分类类型、应用识别或流量分类相对应的预训练神经网络来预测数据包所属的流量类别。下面将解释预处理阶段的数据集、实现和设计细节以及所提出的神经网络的架构。

#### 4.1 数据集

对于这项工作，我们使用“ISCX VPN-nonVPN”流量数据集，该数据集由 pcap 格式文件中被不同应用程序捕获的流量组成（Gil et al. 2016）。在这个数据集中，捕获的数据包被分成不同的 pcap 文件，这些文件根据产生数据包的应用程序（例如，Skype 和 Hangouts 等）和应用程序在捕获会话期间参与的特定活动（例如，语音呼叫、聊天、文件传输或视频通话）。有关捕获的流量和流量生成过程的更多详细信息，请参阅 Gil 等人在 2016 年的工作。

该数据集还包含通过虚拟专用网络 (VPN) 会话捕获的数据包。 VPN 是分布式站点之间的私有覆盖网络，通过公共通信网络（例如英特网）上的隧道流量来运行。隧道 IP 数据包可以保证对服务器和服务的安全远程访问，是 VPN 最突出的方面（Chowdhury 和 Boutaba 2010）。与常规（非 VPN）流量类似，VPN 流量是当执行不同的活动，如语音通话、视频通话和聊天时被不同的应用程序（如 Skype）捕获的。

此外，该数据集包含捕获的 Tor 软件流量。这种流量大概是在使用 Tor 浏览器时产生的，它有 Twitter、Google、Facebook 等标签。Tor 是为匿名通信开发的免费开源软件。 Tor 通过自己的免费全球覆盖网络转发用户流量，该网络由志愿者运营的服务器组成。 Tor 是用来保护用户免受被称为“流量分析”的互联网监视而出现的。为了创建专用网络路径，Tor 通过网络上的中继通过采取一种没有单个中继知道数据包所采用的完整路径的方式建立了一个加密连接链路（Dingledine et al. 2004）。最后，Tor 又使用复杂的端口混淆算法来提高隐私性和匿名性。

#### 4.2 预处理

“ISCX VPN-nonVPN”数据集是在数据链路层捕获的。因此，它包括以太网报头。数据链路报头包含有关物理链路的信息，例如媒体访问控制 (MAC) 地址，这对于在网络中转发帧至关重要，但对于应用程序识别或流量分类任务而言，它是无信息量的。因此，在预处理阶段，首先去除以太网报头。传输层段，特别是传输控制协议 (TCP) 或用户数据报协议 (UDP)，在报头长度上有所不同。前者通常带有 20 字节长度的标头，而后者具有 8 字节的标头。为了使传输层段统一，我们在 UDP 段的头部的末尾注入零，使它们与 TCP 头部的长度相等。然后将数据包从位转换为字节，这有助于减少神经网络的输入大小。

由于数据集是在真实世界的仿真中捕获的，它包含一些不相关的数据包，这些数据包不是我们关心的，应该被丢弃。特别是，数据集包括一些 TCP 段，其中 SYN、ACK 或 FIN 标志设置为 1 并且不包含有效负载。在建立连接或完成一个连接时，三次握手过程需要这些段，但它们不携带有关生成它们的应用程序的信息，因此可以安全地丢弃。此外，数据集中还有一些域名服务 (DNS) 段。这些段用于主机名解析，即将 URL 转换为 IP 地址。这些段与应用程序识别或流量分类无关，因此可以从数据集中删除。
图 4 说明了数据集的数据包长度的直方图（经验分布）。如直方图所示，数据集的数据包长度变化很大，而使用神经网络需要使用固定大小的输入。因此，不可避免地需要做固定长度的截断或零填充。为了找到截断的固定长度，我们检查了数据包长度的统计信息。我们的调查显示，大约 96% 的数据包的有效负载长度小于 1480 字节。这一观察结果与我们的预期相差不远，因为大多数计算机网络都受到 1500 字节的最大传输单元 (MTU) 大小的限制。因此，我们保留 IP 报头和每个 IP 数据包的前 1480 个字节，这会产生一个 1500 个字节的向量作为我们提出的神经网络的输入。 IP 有效负载小于 1480 字节的数据包在末尾补零。为了获得更好的性能，所有数据包字节都除以 255，即一个字节的最大值，因此所有输入值都在 [0, 1] 范围内。

此外，由于神经网络有可能尝试使用 IP 地址来学习对数据包进行分类，因为数据集是使用有限数量的主机和服务器捕获的，因此我们决定通过在 IP 标头屏蔽 IP 地址来防止这种过拟合。在这件事上，我们保证神经网络没有使用不相关的特征来执行分类。上述所有预处理步骤会在当用户将 pcap 文件加载到深度包的工具包中时发生。

##### 4.2.1 标注数据集

如前面4.1节所述，数据集的 pcap 文件根据应用程序和他们从事的活动进行标记。但是，对于应用程序识别和流量分类任务，我们需要对每个任务重新定义标签。对于应用程序识别，在非 VPN 会话期间收集的所有标记为特定应用程序的 pcap 文件都被聚合到一个文件中。这产生表 1a 中显示的 17 个不同的标签。同样对于流量分类，我们将捕获的涉及同一活动的不同应用程序的流量（考虑 VPN 或非 VPN 条件）汇总到单个 pcap 文件中。这产生了一个如表 1b 所示的 12 个类别的数据集。通过观察表 1，我们会立即注意到数据集明显不平衡，并且不同类别之间的样本数量差异很大。众所周知，在训练数据中的这种不平衡会导致分类性能下降。抽样是克服这个问题的一种简单而强大的技术（Longadge 和 Dongre 2013）。因此，为了训练我们提出的神经网络，通过使用欠采样方法，我们随机删除了主要类的样本（具有更多样本的类），直到类相对平衡。

#### 4.3 架构

在下文中，我们将解释我们在 Deep Packet 工具包中使用的两种建议架构。

所提出的 SAE 架构由五个完全连接的层组成，它们相互堆叠，分别由 400、300、200、100 和 50 个神经元组成。为了防止过拟合问题，在每一层之后都采用了具有 0.05 辍学率的辍学技术。在这种技术中，在训练阶段，一些神经元被随机设置为零。因此，在每次迭代中，都有一组随机的活动神经元。对于应用程序识别和流量表征任务，在所提出的 SAE 的最后一层，分别添加了一个具有 17 个和 12 个神经元的 softmax 分类器。

图 5 描绘了基于一维 (1D) CNN 的第二种提议方案的最小图示。我们在超参数空间的子空间上使用网格搜索来选择能够产生最佳性能的方案.此过程在第 3 节中详细讨论。 5. 我们最终提出的模型由两个连续的卷积层组成，然后是一个池化层。然后，将二维张量压缩成一维向量，并馈入一个由全连接神经元组成的三层网络，该网络也采用 dropout 技术来避免过拟合。最后，将 softmax 分类器应用于分类任务，类似于 SAE 架构。为超参数找到的最佳值如表 2 所示。用于应用识别和流量表征任务的所有建议模型的详细架构可在“附录 A”中找到。

### 5 实验结果

为了实现我们提出的 NN，我们使用了 Keras 库 (Chollet et al. 2017)，并以 Tensorflow (Abadi et al. 2015) 作为其后端。每个提出的模型都针对从数据集中提取的独立测试集进行了训练和评估。我们将数据集随机分成三个独立的集合。第一个包含 64% 的样本用于训练和调整权重和偏差。第二部分包含 16% 的样本用于训练阶段的验证，最后由 20% 的数据点组成的第三部分用于测试模型。此外，为了避免过拟合问题，我们使用了早期停止技术（Prechelt 1998）。一旦验证集上的损失函数值在几个时期内几乎保持不变，这种技术就会停止训练过程，从而防止网络过度拟合训练数据。为了加快学习阶段，我们还在我们的模型中使用了批量标准化技术（Ioffe 和 Szegedy 2015）。

为了训练 SAE，首先使用 Adam 优化器 (Kingma and Ba 2014) 以贪婪的逐层方式训练每一层，并将均方误差作为 200 个 epoch 的损失函数，如 Sect 中所述。 3.1。接下来，在微调阶段，使用分类交叉熵损失函数对整个网络进行另外 200 个 epoch 的训练。此外，为了实现所提出的一维 CNN，分类交叉熵和 Adam 分别用作损失函数和优化器，在这种情况下，网络训练了 300 个 epoch。最后，值得一提的是，在两个 NN 中，除了最终的 softmax 分类器层外，所有层都使用 Rectified Linear Unit (ReLU) 作为激活函数。

为了评估 Deep Packet 的性能，我们使用了 Recall (Rc)、Precision (Pr) 和 F1 Score（即 F1）指标。上述指标在数学上描述如下

$$功能$$

其中 TP、FP 和 FN 分别代表真阳性、假阳性和假阴性。

正如教派中提到的那样。 4，我们使用网格搜索超参数调整方案来找到我们工作中最好的 1D-CNN 结构。由于我们的计算硬件限制，我们只搜索了超参数的受限子空间 5 实验结果，以找到最大化每个任务的测试集上的加权平均 F1 分数的结果。更具体地说，我们更改了两个卷积层的过滤器大小、过滤器数量和步幅。总共评估了 116 个模型，它们在应用程序识别和流量表征任务中的加权平均 F1 得分。所有训练模型的结果如图 6 所示。我们认为无法为流量分类任务选择最佳模型，因为“最佳模型”的定义没有明确定义，并且模型准确性和模型精度之间存在权衡。它的复杂性（即训练和测试速度）。在图 6 中，每个点的颜色与模型的可训练参数相关联；颜色越深，可训练参数的数量就越多。

如图 6 所示，增加神经网络的复杂性并不一定会带来更好的性能。导致这种现象的原因有很多，其中可以提到梯度消失和过拟合问题。一个复杂的模型更容易面临梯度消失的问题，这会导致训练阶段的欠拟合。另一方面，如果学习模型变得更复杂，而训练数据的大小保持不变，就会出现过拟合问题。这两个问题都导致NN在评估阶段的表现不佳。

表 3 显示了 SAE 和 1D-CNN 在测试集上的应用识别任务所取得的性能。 1D-CNN 和 SAE 的加权平均 F1 分数分别为 0.98 和 0.95，表明我们的网络已经完全从训练集中提取和学习了区分特征，并且可以成功区分每个应用程序。对于流量表征任务，我们提出的 CNN 和 SAE 分别取得了 0.93 和 0.92 的 F1 分数，这意味着这两个网络都能够准确地对数据包进行分类。表 4 总结了所提出的方法在测试集上实现的性能。
 
#### 5.1 比较

下面，我们将 Deep Packet 的结果与之前使用“ISCX VPN-nonVPN”数据集的结果进行比较。此外，Deep Packet 与 Sect 中的其他一些机器学习方法进行了比较。 5.1.2.

##### 5.1.1 与以往结果的比较

正如教派中提到的那样。 2，Gil 等人的作者。 (2016) 尝试使用从流量流中手工制作的时间相关特征来表征网络流量，例如流的持续时间和每秒的流字节数。亚曼萨瓦斯拉尔等人。还使用此类与时间相关的特征来识别最终用户应用程序（Yamansavascilar et al. 2017）。这两项研究都在“ISCX VPN-nonVPN 流量数据集”上评估了他们的模型，其最佳结果见表 5。结果表明，Deep Packet 在应用程序识别和流量表征方面都优于上述其他提议的方法任务。

我们要强调的是，上述工作使用了基于网络流量的手工特征。另一方面，Deep Packet 在数据包级别考虑网络流量，并且可以对网络流量的每个数据包进行分类，这是一项更难的任务，因为与单个数据包相比，流中的信息更多。此功能使 Deep Packet 更适用于实际情况。

最后，值得一提的是，与我们的工作（Lotfollahi et al. 2017）独立且平行，Wang et al.提出了一种与 Deep Packet 类似的方法，用于对“ISCX VPN-nonVPN”流量数据集进行流量表征（Wang et al. 2017）。他们报告的最佳结果在流量表征任务上实现了 100% 的精度。然而，我们认为他们的结果存在严重问题。我们指控的证明理由是，他们的最佳结果是通过使用包含来自互联网协议栈每五层的所有标头的数据包获得的。然而，根据我们的实验以及数据集提供者的直接询问（Gil et al. 2016），在“ISCX VPN-nonVPN”流量数据集中，源和目标 IP 地址（出现在网络层的标头中）每个应用程序都是独一无二的。因此，他们的模型可能只是使用此功能对流量进行分类（在这种情况下，更简单的分类器就足以处理分类任务）。如前所述，为了避免这种现象，我们在预处理阶段屏蔽 IP 地址字段，然后将数据包送入我们的 NN 进行训练或测试。

##### 5.1.2 与以往方法的比较

在本节中，我们将 Deep Packet 与四种机器学习算法进行比较。通过提供类似于我们提供给 Deep 数据包的预先拥有的数据包来进行比较。我们使用 scikit-learn (Pedregosa et al. 2011) 实现深度为 2 的决策树、深度为 4 的随机森林、逻辑回归（c = 0.1）和具有默认参数的朴素贝叶斯。表 6 表明我们的方法在测试数据的应用识别任务中优于四种替代算法。同样，表 7 说明 Deep Packet 在流量表征任务中表现更好。

这些比较证实了深度神经网络在需要分析大量数据的网络流量分类中的强大功能。

### 6 讨论

在应用程序识别和流量表征任务的测试集上评估 SAE 会产生如图 7 所示的行归一化混淆矩阵。混淆矩阵的行对应于样本的实际类别，列表示预测标签;因此，矩阵是行归一化的。主对角线上元素的深色表明 SAE 可以对每个应用程序进行分类，但会产生轻微的混淆。

通过仔细观察图 7 中的混淆矩阵，人们会注意到不同类别（例如 ICQ 和 AIM）之间的一些有趣的混淆。层次聚类进一步证明了 Deep Packet 捕获的相似性。使用 SAE 对行归一化混淆矩阵进行聚类以进行应用程序识别（图 7a），使用欧几里德距离作为距离度量，使用 Ward.D 作为聚集方法，揭示了应用程序之间在分配给 17 个应用程序类的倾向方面的相似性。如图 8a 所示，Deep Packet 揭示的应用程序分组通常与现实世界中应用程序的相似性一致。层次聚类将应用程序分为 7 组。有趣的是，这些组在某种程度上类似于流量表征任务中的组。人们会注意到捆绑在一起的 Vimeo、Netflix、YouTube 和 Spotify 都是流媒体应用程序。还有一个集群，包括ICQ、AIM和Gmail。 AIM 和 ICQ 用于在线聊天，Gmail 除了电子邮件服务外，还提供在线聊天服务。另一个有趣的观察结果是 Skype、Facebook 和 Hangouts 都组合在一个集群中。尽管这些应用程序似乎不太相关，但这种分组是合理的。该数据集以三种形式包含这些应用程序的流量：语音通话、视频通话和聊天。因此，网络发现这些应用程序在使用方面相似。用于在两个远程系统之间安全传输文件的 FTPS（基于 SSL 的文件传输协议）和 SFTP（基于 SSH 的文件传输协议）也被聚集在一起。有趣的是，SCP（安全副本）已经形成了它的集群，尽管它也用于远程文件传输。 SCP 使用 SSH 协议传输文件，而 SFTP 和 FTPS 使用 FTP。据推测，我们的网络已经了解了这种细微的差异并将它们分开。 Tor 和 Torrent 的集群是明智的，因为它们与其他应用程序存在明显差异。这种聚类并非完美无缺。将 Skype、Facebook 和环聊与电子邮件和 VoipBuster 一起集群是不正确的。 VoipBuster 是一个通过 Internet 基础设施提供语音通信的应用程序。因此，该集群中的应用程序在使用方面似乎不太相似，而且这种分组并不精确。

对流量特征的混淆矩阵执行相同的过程，如图 8b 所示。有趣的是，分组将流量分成 VPN 和非 VPN 集群。所有 VPN 流量都捆绑在一个集群中，而所有非 VPN 流量都组合在一起。

正如教派中提到的那样。 2，许多应用程序使用加密来维护客户的隐私。结果，大多数“ISCX VPN-nonVPN”数据集流量也被加密了。有人可能想知道 Deep Packet 如何对此类加密流量进行分类。与 DPI 方法不同，Deep Packet 不检查数据包中的关键字。相反，它尝试学习每个应用程序生成的流量中的特征。因此，它不需要解密数据包来对它们进行分类。

理想的加密方案会导致输出消息承载最大可能的熵（Cover 和 Thomas 2006）。换句话说，它产生了理论上无法相互区分的无模式数据。然而，由于所有实际的加密方案都使用伪随机生成器，这个假设在实践中是不成立的。此外，每个应用程序都采用不同的（非理想）加密方案进行数据加密。这些方案使用不同的伪随机生成器算法，从而产生可区分的模式。这种模式的变化可用于将应用程序彼此分开。 Deep Packet 尝试提取这些判别模式并学习它们。因此，它可以准确地对加密流量进行分类。

从表 3 可以看出，Tor 流量也被成功分类。为了进一步研究这种流量，我们进行了另一项实验，在该实验中，我们使用仅包含 Tor 流量的数据集训练和测试了 Deep Packet。如前所述，为了获得最佳结果，我们对 NN 的超参数进行了网格搜索。详细结果见表 8，表明 Deep Packet 无法准确分类底层 Tor 的流量。这种现象与我们的预期相去不远。 Tor 在传输之前对其流量进行加密。如前所述，Deep Packet 可能会学习应用程序使用的各种加密方案中使用的不同伪随机模式。在这个实验中，流量通过 Tor 进行隧道传输。因此，它们都经历相同的加密方案。因此，我们的神经网络无法很好地将它们分开。

### 7 未来的工作

深度神经网络在实践中表现如此出色的原因尚不清楚。此外，没有严格的理论框架来设计和分析此类网络。如果在这些方面取得一些进展，将对提出更好的专门用于网络流量分类的深度神经网络结构产生直接影响。同样，未来另一个重要的方向之一是研究我们提出的模型的可解释性（Du et al. 2018; Montavon et al. 2018; Samek et al. 2018）。这将包括分析模型已经学习的特征以及学习它们的过程。

另一个需要研究的重要方向是使用对抗性攻击算法对所提出的方案针对噪声和恶意生成的输入进行鲁棒性分析（Yuan et al. 2017）。针对机器学习方法的对抗性攻击已在其他一些领域（例如，Akhtar 和 Mian 2018；Huang 等人 2017；Carlini 和 Wagner 2018）得到广泛研究，但在网络流量分类中却没有。

设计多级分类算法也是未来研究的一个有趣的可能方向。这意味着系统应该能够检测流量是来自已知的先前类别之一还是新的“未知”类别。如果数据包被标记为未知，那么它将被添加到未知类的数据库中。此外，通过接收更多未知数据包，可以使用无监督聚类算法将它们标记为离散类。接下来，人类专家将能够将这些未知类映射到众所周知的现实世界应用程序。因此，使用这些新的标记类重新训练第一级分类器将成为可能。可以使用在线学习算法或使用先前学习的神经网络权重作为新网络的初始化来完成重新训练。

最后，实施所提出的方案以能够处理现实世界的高速网络流量将是一个重要的现实挑战。这可以通过利用硬件实现（例如，参见 Vanhoucke 等人 2011；Zhang 等人 2015）和应用神经网络简化技术（例如，参见 Hubara 等人 2017；Lin 等人 2016）来完成.

### 8 结论

在本文中，我们提出了 Deep Packet，这是一个使用深度学习算法自动从计算机网络流量中提取特征以对流量进行分类的框架。据我们所知，Deep Packet 是第一个使用深度学习算法的流量分类系统，即 SAE 和 1D-CNN，可以处理应用程序识别和流量表征任务。我们的结果表明，迄今为止，在应用程序识别和流量表征任务中，Deep Packet 优于“ISCX VPN-nonVPN”流量数据集上的所有类似工作。此外，凭借 Deep Packet 取得的最新成果，我们设想 Deep Packet 是朝着在流量分类和更普遍的网络分析任务中使用深度学习算法的总体趋势迈出的第一步。此外，可以修改 Deep Packet 以处理更复杂的任务，例如多通道（例如，区分不同类型的 Skype 流量，包括聊天、语音通话和视频通话）分类、Tor 流量的准确分类等。最后，自动从网络流量中提取特征可以节省聘请专家从流量中识别和提取手工特征的成本，最终导致更准确的流量分类。