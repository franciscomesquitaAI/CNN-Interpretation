# Motivation

Convolutional Neural Networks (CNNs) have revolutionized computer vision tasks, but their complex decision-making processes often obscure understanding. This opacity raises concerns about reliability and trustworthiness, particularly in critical domains. Interpretation techniques for CNNs are essential for addressing these concerns, as they provide insights into model predictions, enhancing transparency and accountability \[1-3].

This work aims to explore a range of interpretation techniques tailored for CNNs to achieve several objectives:

- **Enhancing Transparency:** By revealing the features driving CNN predictions, interpretation techniques make CNNs more understandable to users, resulting in more trust in AI systems.

- **Debugging and Error Analysis:** These techniques facilitate error diagnosis and identify sources of misclassification, enabling targeted improvements to enhance model performance and robustness.

- **Ensuring Accountability and Fairness:** Interpretation techniques empower stakeholders to scrutinize and validate CNN decisions, promoting accountability and mitigating risks of bias or discrimination.

- **Facilitating Domain-Specific Insights:** Through empirical evaluation, we seek to provide actionable insights and recommendations for leveraging CNNs effectively in diverse real-world scenarios.

The techniques under investigation encompass gradient-based attribution methods like GradCAM and Guided Backpropagation, perturbation-based approaches such as LIME and Occlusion Sensitivity, and advanced techniques like RISE, Saliency Maps, Anchor Explanations, and Activation Maximization.

---
# Dataset

This was above all a work of analyzing, understanding and comparing the different interpretability methods, so the data used was chosen purely for proof of concept. 

The Monkeypox Image Dataset is a collection of images representing six distinct classes related to infectious diseases: Chickenpox, Cowpox, Measles, Monkeypox, Smallpox or Healthy. Each class represents different conditions or diseases that can manifest visually on the skin, with varying levels of severity and characteristics.

- **Chickenpox:** Images in this class depict the characteristic rash and blisters associated with chickenpox, caused by the varicella-zoster virus. The rash typically starts on the chest, back, and face before spreading to other parts of the body.

- **Cowpox:** This class contains images showing the skin lesions caused by cowpox virus infection. Cowpox lesions are typically localized to the site of viral entry, often on the hands or face, and can resemble small, red blisters or ulcers.

- **Healthy:** Images in this class serve as a baseline for comparison, representing individuals with no visible skin abnormalities or signs of disease. These images may include various skin types and colors to ensure diversity within the dataset.

- **Measles:** Measles images depict the characteristic rash associated with measles virus infection. The rash typically appears as flat, red spots that often merge together, covering large areas of the body.

- **Monkeypox:** Images in this class showcase the clinical manifestations of monkeypox virus infection. Monkeypox lesions are similar in appearance to smallpox but are typically less severe. They begin as raised bumps and progress to fluid-filled blisters, which can lead to scarring.

- **Smallpox:** This class contains images depicting the severe skin manifestations caused by the variola virus, which causes smallpox. Smallpox lesions are typically deeply embedded in the skin and progress through stages of papules, vesicles, pustules, and scabs.

![images classes](https://i.imgur.com/PBgQ2LM.png)

---

# Transfer Learning on Xception model

CNN training and performance are not the focus of this work and so, whenever possible, I used the Xception network previously trained on the imagenet dataset \[4].

![Transfer learning concept](https://i.imgur.com/vvKaDxV.png)

The following techniques to interpret the CNN were used and will be explained below:
1. GradCAM
2. LIME
3. RISE
4. Saliency Maps
5. Anchor Explanations
6. Activation Maximization
7. Occlusion Sensitivity
8. Guided Backpropagation
9. Deep Dream

---
# GradCAM

Based on works \[5-7].

Grad-CAM is a popular technique for visualizing where a convolutional neural network model is looking. Grad-CAM is class-specific, meaning it can produce a separate visualization for every class present in the image:

![GradCAM example](https://i.imgur.com/dfo7jxS.png)

Grad-CAM can be applied to networks with general CNN architectures, containing multiple fully connected layers at the output. It extends the applicability of the CAM procedure by incorporating gradient information. Specifically, the gradient of the loss. The last convolutional layer determines the weight for each of its feature maps. As in the CAM procedure above, the further steps are to compute the weighted sum of the activations and then upsampling the result to the image size to plot the original image with the obtained heatmap.

![GradCAM algorithm logic](https://i.imgur.com/ZoeRFhW.png)

A significant drawback of this procedure is that it requires the network to use global average pooling (GAP) as the last step before the prediction layer. It thus is not possible to apply this approach to general CNNs. An example is shown in the figure below:

![Need class activation mapping](https://i.imgur.com/fMV2Q3M.png)

While the Grad-CAM paper has garnered thousands of citations, recent studies have unveiled a significant issue with its methodology: Grad-CAM occasionally highlights regions within an image that the model did not employ in its prediction process. This inconsistency raises doubts about the reliability of Grad-CAM as an explanation method for models. Addressing this concern, [HiResCAM](https://arxiv.org/abs/2011.08891) emerges as a novel explanation technique that offers a provable guarantee to exclusively highlight locations utilized by the model. Inspired by Grad-CAM, HiResCAM represents a promising advancement in the realm of model interpretation.

Although there are problems on Grad-CAM it is the most used variation of CAM but there are a lot of variation from the original CAM algorithm. Here, we will use the GradCAM one because it is the mostly used globally.

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-gradcam.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-gradcam)

Original image to be predicted by the model:

![GradCAM originalImg](https://i.imgur.com/hzbEYgk.jpeg)

Heatmap array generated by GradCAM:
![GradCAM heatmapArray](https://i.imgur.com/PMdUYsw.png)

Output with GradCAM generated interpretability:
![GradCAM generatedInterpretability](https://i.imgur.com/yBNHz8N.png)

---
# LIME

Based on works \[8-10]

LIME stands for Local Interpretable Model-agnostic Explanations. It is a Python library based on a paper from Ribeiro et al. \[11] to help you understand the behavior of your black-box classifier model. Currently, you can use LIME for a classifier model that classify tabular data, images, or texts.

The abbreviation of LIME itself should give you an intuition about the core idea behind it. LIME is:
- **Local**, which means that LIME tries to find the explanation of your black-box model by approximating the local linear behavior of your model.
- **Interpretable**, which means that LIME provides you a solution to understand why your model behaves the way it does.
- **Model agnostic**, which means that LIME is model-independent. In other words, LIME is able to explain any black-box classifier you can think of.

**How it really works?**

Internally, LIME tries to interpret a black box model by conducting these four steps:

1. **Input data permutation:** The first step that LIME would do is to create several artificial data points that are close with the data denoted by the red star.

![LIME inputDataPermutations](https://i.imgur.com/aeoUzeh.png)

If our input data is an image, LIME will generate several samples that are similar with our input image by turning on and off some of the super-pixels of the image (A superpixel can be defined as a group of pixels that share common characteristics like pixel intensity).

![LIME SimilarSamplesGenerated](https://i.imgur.com/rbqYWbr.png)

2. **Predict the class of each artificial data point:** Next, LIME will predict the class of each of the artificial data point that has been generated using our trained model. If your input data is an image, then the prediction of each perturbed image will be generated at this stage.

3. **Calculate the weight of each artificial data point:** The third step is to calculate the weight of each artificial data to measure its importance. To do this, first the cosine distance metric is usually applied to calculate how far the distance of each artificial data point with respect to our original input data. Next, the distance will be mapped into a value between zero to one with a kernel function. The closer the distance, the closer the mapped value to one, and hence, the bigger the weight. The bigger the weight, the bigger the importance of a certain artificial data point. If the input data is an image, then the cosine distance between each perturbed image and the original image will be computed. The more the similarity between a perturbed image to the original image, the bigger its weight and importance.

4. **Fit a linear classifier to explain the most important features:** The last step is fitting a linear regression model using the weighted artificial data points. After this step, we should get the fitted coefficient of each feature, just like the usual linear regression analysis. Now if we sort the coefficient, the features that have larger coefficients are the ones that play a big role in determining the prediction of our black-box machine learning model.

Imagine we want to explain a classifier that predicts how likely it is for the image to contain a tree frog. We take the image on the left and divide it into interpretable components (contiguous superpixels).

![LIME PraticalExample](https://i.imgur.com/lvwJsNF.png)

We then generate a data set of perturbed instances by turning some of the interpretable components “off” (in this case, making them gray). For each perturbed instance, we get the probability that a tree frog is in the image according to the model. We then learn a simple (linear) model on this data set, which is locally weighted—that is, we care more about making mistakes in perturbed instances that are more similar to the original image. In the end, we present the superpixels with highest positive weights as an explanation, graying out everything else.

![LIME AlgorithmOverview](https://i.imgur.com/1stZHfp.png)

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-lime.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-lime)

Example of LIME in this work:

![LIME Results obtained](https://i.imgur.com/WOqZczN.png)

---
# RISE - Randomized Image Sampling for Explanations

Based on works \[12-14]

RISE queries black-box model on multiple randomly masked versions of input. After all the queries are done we average all the masks with respect to their scores to produce the final saliency map. The idea behind this is that whenever a mask preserves important parts of the image it gets higher score, and consequently has a higher weight in the sum.

![RISE UsageExample](https://i.imgur.com/kaVn9qa.png)

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-rise.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-rise)(Here, only Kaggle has the outputs for each cell)

![RISE Implementation 1](https://i.imgur.com/Al4C5g8.png)

**It seems less accurate than the GradCAM algorithm but we need different, more, and (better?) data to validate that**

---
# Saliency Maps

Based on works \[15-17].

Saliency maps get a step further by providing an interpretable technique to investigate hidden layers in CNNs. It is the oldest and most frequently used explanation method for interpreting the predictions of convolutional neural networks. The saliency map is built using gradients of the output over the input. This highlights the areas of the images which were relevant for the classification.

Images are processed using saliency maps to distinguish visual features. Colored photos, for example, are converted to black-and-white pictures so that the strongest colors can be identified. Two other examples are the infrared to detect temperature (red is hot, blue is cold) and the night vision to identify light sources (green is bright and black is dark).

For example, we can see below that water plays a significant role when recognizing a ship. Maybe the model won’t be so successful if it is given a ship outside water in a construction site. This observation provides important clues about the need to retrain the model with additional images of ships in different environmental conditions.

![SaliencyMaps firstexample](https://i.imgur.com/QC3iyvN.png)

There is a lot techniques based on saliency maps but in this notebook we focus on the Vanilla Gradient:
- Forward pass with data
- Backward pass to input layer to get the gradient
- Render the gradient as a normalized heatmap

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-saliency-maps.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-saliency-maps).

This method required that images were normalized:

![Saliency NormalvsNormalized](https://i.imgur.com/XRjOmJV.png)

Final result of this interpretation technique:

![Saliency finalResult](https://i.imgur.com/gybbni4.png)

---
# Anchor Explanations

It was implemented with the ALIBI Explain framework \[18]. The work was based on \[19-21].

Similar to LIME, images are first segmented into superpixels, maintaining local image structure. The interpretable representation then consists of the presence or absence of each superpixel in the anchor. It is crucial to generate meaningful superpixels in order to arrive at interpretable explanations. The algorithm supports a number of standard image segmentation algorithms (felzenszwalb, slic and quickshift) and allows the user to provide a custom segmentation function.

The superpixels not present in a candidate anchor can be masked in 2 ways:
- Take the average value of that superpixel.
- Use the pixel values of a superimposed picture over the masked superpixels.

![AnchorExplanations generaldea](https://i.imgur.com/XukubXk.png)

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-anchor-explanations.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-anchor-explanations).

![AnchorsExplanation implementation](https://i.imgur.com/i0B9mBZ.png)

---
# Activation Maximization

Based on the works \[22-24].

**This is a technique to interpret the network from inside and not the output itself.**

Activation maximization, as the name indicates, aims to maximize the activation of certain neurons. Imagine you are training your model with a single image several times. Training is changing the weights accordingly to achieve the lowest loss possible, so the input and the desired output will be constant whereas the weights will be modified iteratively until we reach a minima (or until we decide to stop training). **In Activation Maximization, we will keep the weights and the desired output constant and we will modify the input such that it maximizes certain neurons.**

A network’s activation function output represents how confident it is that a training example belongs to one specific class, and so activation maximization constructs an image that checks off every single box the neural network is looking for and hence yields the largest activation function. This is done with gradient ascent, which tries to maximize the output neuron. **The idea of activation maximization is really just finding the inputs that return an output with the highest confidence.**

![ActivationMaximization example](https://i.imgur.com/RYviI44.png)

The results are really a quite enlightening vision into how the model makes decisions; dark regions represent ‘penalizations’ in that high values in that region make the model less sure the input is that digit, and bright values represent ‘bonuses’ in that high values in those regions increase the confidence of the output neuron. Activation maximization can also be visualized in the form of a distribution or another distribution representation for one-dimensional, non-image data.

**Gradient Ascent algorithm** (Also used on Deep Dream technique)

When creating DeepDream images, the idea is to modify input image so as to maximize the activation of a specific feature map or maps. It's like asking the neural network what it wants to see as input. Neural network itself guides the input modification process.

While training neural networks, we aim to decrease loss and use gradient descent for this purpose. To maximize an activation, we use gradient ascent.

The gradient ascent method advances in the direction of the gradient at each step. The gradient is assessed beginning at point P0, and the function proceeds to the next point, P1. The function then advances to P2 when the gradient is reevaluated at P1. This loop will continue until a stopping condition is fulfilled. The gradient operator always ensures that we are travelling in the best direction feasible.

![GradientAscentAlgo](https://i.imgur.com/buEJ1TM.png)

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-activation-maximization.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-activation-maximization).

See below the technique applied to the data used:

![ActivationMaximization owndata](https://i.imgur.com/NL3KDLn.jpeg)

**It is almost impossible to interpret what are the patterns that the network look on this data. Maybe in some data this method could be more useful**

It would be more useful if we could see the filters that contributed most to a particular ranking. I will look into how this can be done. This in theory can be solved with this technique (Filter Maximization) but applied in a different way.

---
# Occlusion Sensitivity

Based on the works \[25-26].

The basic concept is as simple as they come: For every input dimension of an input x, we evaluate the model with that dimension missing, and observe how the output changes. In particular, if $||f(x) — f(x-i)||$ is large, then the dimension must have been important because removing it changes the output a lot.

![Occlusion explanation](https://i.imgur.com/kNIUVw5.png)

Unfortunately, in most cases, such as image data, this is not the case. Here, you would be advised to remove whole patches instead of individual pixels. The idea is that usually the information of a single pixel can be reconstructed from its neighbors. So **if you have an image of a cat, removing one cat-pixel will never have a large effect on the output, whereas removing the patch covering an ear might lead to a noticeable drop in the model’s prediction for ‘cat’**.

As long as you can feed in inputs and receive outputs, you can use occlusion analysis. In some cases this could be used to remove noise of an image and improve classification confidence.

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-occlusion-sensitivity.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-occlusion-sensitivity).

Final result:

![Occlusion results](https://i.imgur.com/sgEeYc6.png)

---
# Guided Backpropagation

Based on the works \[27 - 29]

Guided Backpropagation (GBP) is an approach designed by Springenberg et al., relying on the ideas of Deconvolution and Saliency. Authors argue that the approach taken by Simonyan et al. with the saliency maps has an issue with the flow of negative gradients, which decreases the accuracy of the higher layers we are trying to visualize. Their idea is to combine two approaches and add a “guide” to the Saliency with the help of deconvolution.

Guided Backpropagation combines vanilla backpropagation at ReLUs (leveraging which elements are positive in the preceding feature map) with DeconvNets (keeping only positive error signals). We are only interested in what image features the neuron detects. So when propagating the gradient, we set all the negative gradients to 0. We don’t care if a pixel “suppresses’’ (negative value) a neuron somewhere along the part to our neuron. Value in the filter map greater than zero signifies the pixel importance which is overlapped with the input image to show which pixel from the input image contributed the most.

**Given below is the example of how guided backpropagation works:**

- Relu Forward pass

![Guided relu forward pass](https://i.imgur.com/P8H7e23.png)

- Relu Backward Pass ( flow the value as it is where value is greater than zero in the filter (h_l) during forward propagation.)

![relu backwardpass](https://i.imgur.com/rDh44vH.png)

- Deconvolution for Relu - flow the values backward as it is where value in the filter is greater than 0.

![relu deconvulotion](https://i.imgur.com/QaXNIZB.png)

- **Guided Backpropagation** - taking the intersection of the concept of Backward pass and the deconvolution.

![guided backpropagation](https://i.imgur.com/DtOSass.png)

**Final Result:**

![guided backpropagation example](https://i.imgur.com/rqGa8HO.png)

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-guided-backpropagation.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-guided-backpropagation).

Result obtained:

![guided backpropagation result](https://i.imgur.com/p9e7DTm.png)

---
# Deep Dream

Based on the works \[30-34]

DeepDream is an experiment that visualizes the patterns learned by a neural network. Similar to when a child watches clouds and tries to interpret random shapes, DeepDream over-interprets and enhances the patterns it sees in an image. It does so by forwarding an image through the network, then calculating the gradient of the image with respect to the activations of a particular layer. The image is then modified to increase these activations, enhancing the patterns seen by the network, and resulting in a dream-like image.

It is almost only used in generative art but i do think that it can be used to interpret a network: To fully understand a network prediction, we have to understand why it gives the output but also what the network sees from inside (patterns learned). The second part is what deep dream does in an somewhat abastract and over exaggerated way. It combines the input image with the activations of a particular layer producing a dream like image. This dream like image has a lot of usefull information. It is not in a direct and very perceptible way, but we can see the patterns learned by the network for each class through these images.

There is a demonstration in real time of what deep dream is: [https://www.youtube.com/watch?v=DgPaCWJL7XI](https://www.youtube.com/watch?v=DgPaCWJL7XI)

We can see very dog faces and eyes. That is because the original Deep Dream network was trained on ImageNet dataset that as a lot of dogs in it. The network is just reproducing the patterns that it has learned. **If we use our network instead of that trained on imagenet, we can see the patterns learned by the network.** Although it is very popular in generative and artificial art i could not found any other use for this concept.

**Gradient Ascent**

When creating DeepDream images, the idea is to modify input image so as to maximize the activation of a specific feature map or maps. It's like asking the neural network what it wants to see as input. Neural network itself guides the input modification process.

While training neural networks, we aim to decrease loss and use gradient descent for this purpose. To maximize an activation, we use gradient ascent.

The gradient ascent method advances in the direction of the gradient at each step. The gradient is assessed beginning at point P0, and the function proceeds to the next point, P1. The function then advances to P2 when the gradient is reevaluated at P1. This loop will continue until a stopping condition is fulfilled. The gradient operator always ensures that we are travelling in the best direction feasible.

## Implementation

See the full implementation at [GitHub](https://github.com/franciscomesquitaAI/CNN-Interpretation/blob/main/cnn-interpretation-deep-dream.ipynb) or at [Kaggle](https://www.kaggle.com/code/franciscomesquita/cnn-interpretation-deep-dream).

![DeepDream result](https://i.imgur.com/7yAFoDy.jpeg)


**All the different implementations are present on the GitHub repository:** [https://github.com/franciscomesquitaAI/CNN-Interpretation](https://github.com/franciscomesquitaAI/CNN-Interpretation)

---
# Conclusion

This work is an in-depth study of interpretability algorithms for CNNs. A total of 9 different algorithms were tested. Despite Deep Dream not being traditionally categorized as an interpretability algorithm, its intriguing potential was the reason to include it in the study. Moving forward, additional experiments are planned to further elucidate remaining ambiguities and delve into unexplored facets of all these methods capabilities.

## Strengths
- Overall, i have looked into and put into practice 9 different interpretability techniques for convolutional neural networks that have been popular in recent times.
- This work serves as a basis for further research in the area of explainable artificial intelligence, more closely linked to computer vision and CNNs.
- Various interpretability techniques are introduced here, each with its own origins. Some aim to reveal the reasoning behind the model's output, while others focus on interpreting the internal representations of the network and how it perceives specific objects or classes. This distinction is both intriguing and significant.
## Limitations
- The neural network used was trained on unbalanced data and this can cause some problems. However, this was done on purpose to see if any of the methods would indicate such an issue.
- I would have liked to implement the Deep SHAP method \[35], which is the SHAP method (SHapley Additive exPlanations) but geared towards deep learning. This was not possible due to some technical difficulties and incompatibilities.
## Future work
- Implementation of different interpretability methods and their variations.
- Comparison of methods, grouping methods by their operating logic, and even a comparison within those that propose to do something similar.
- The ideal way to conclude and build on this work would be to carry out a full analysis of the existing literature on the interpretability of CNNs and write a comparative analysis. 

---
# References

\[1]: [https://dl.acm.org/doi/full/10.1145/3563691](https://dl.acm.org/doi/full/10.1145/3563691)

\[2]: [https://www.sciencedirect.com/science/article/pii/S014193822200066X](https://www.sciencedirect.com/science/article/pii/S014193822200066X)

\[3]: [https://www.sciencedirect.com/science/article/pii/S0010482521003723](https://www.sciencedirect.com/science/article/pii/S0010482521003723)

\[4]: [https://arxiv.org/abs/1610.02357v3](https://arxiv.org/abs/1610.02357v3)

\[5]: [https://towardsdatascience.com/understand-your-algorithm-with-grad-cam-d3b62fce353](https://towardsdatascience.com/understand-your-algorithm-with-grad-cam-d3b62fce353)

\[6]: [https://glassboxmedicine.com/2020/05/29/grad-cam-visual-explanations-from-deep-networks/](https://glassboxmedicine.com/2020/05/29/grad-cam-visual-explanations-from-deep-networks/)

\[7]: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

\[8]: [https://towardsdatascience.com/interpreting-image-classification-model-with-lime-1e7064a2f2e5](https://towardsdatascience.com/interpreting-image-classification-model-with-lime-1e7064a2f2e5)

\[9]: [https://darshita1405.medium.com/superpixels-and-slic-6b2d8a6e4f08](https://darshita1405.medium.com/superpixels-and-slic-6b2d8a6e4f08)

\[10]: [https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/](https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/)

\[11]: [https://arxiv.org/abs/1602.04938](https://arxiv.org/abs/1602.04938)

\[12]: [https://arxiv.org/pdf/1806.07421.pdf](https://arxiv.org/pdf/1806.07421.pdf)

\[13]: [https://www.researchgate.net/publication/325893765_RISE_Randomized_Input_Sampling_for_Explanation_of_Black-box_Models](https://www.researchgate.net/publication/325893765_RISE_Randomized_Input_Sampling_for_Explanation_of_Black-box_Models)

\[14]: [https://github.com/eclique/RISE](https://github.com/eclique/RISE)

\[15]: [https://towardsdatascience.com/practical-guide-for-visualizing-cnns-using-saliency-maps-4d1c2e13aeca](https://towardsdatascience.com/practical-guide-for-visualizing-cnns-using-saliency-maps-4d1c2e13aeca)

\[16]: [https://www.marktechpost.com/2022/03/07/an-introduction-to-saliency-maps-in-deep-learning/](https://www.marktechpost.com/2022/03/07/an-introduction-to-saliency-maps-in-deep-learning/)

\[17]: [https://andrewschrbr.medium.com/saliency-maps-for-deep-learning-part-1-vanilla-gradient-1d0665de3284](https://andrewschrbr.medium.com/saliency-maps-for-deep-learning-part-1-vanilla-gradient-1d0665de3284)

\[18]: [https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_image_imagenet.html](https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_image_imagenet.html)

\[19]: [https://www.kaggle.com/general/226091](https://www.kaggle.com/general/226091)

\[20]: [https://homes.cs.washington.edu/~marcotcr/aaai18.pdf](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)

\[21]: [https://github.com/SeldonIO/alibi/blob/master/doc/source/methods/Anchors.ipynb](https://github.com/SeldonIO/alibi/blob/master/doc/source/methods/Anchors.ipynb)

\[22]: [http://laid.delanover.com/introduction-to-activation-maximization-and-implementation-in-tensorflow/](http://laid.delanover.com/introduction-to-activation-maximization-and-implementation-in-tensorflow/)

\[23]: [https://towardsdatascience.com/every-ml-engineer-needs-to-know-neural-network-interpretability-afea2ac0824e](https://towardsdatascience.com/every-ml-engineer-needs-to-know-neural-network-interpretability-afea2ac0824e)

\[24]: [https://arxiv.org/pdf/1904.08939.pdf](https://arxiv.org/pdf/1904.08939.pdf)

\[25]: [https://towardsdatascience.com/inshort-occlusion-analysis-for-explaining-dnns-d0ad3af9aeb6](https://towardsdatascience.com/inshort-occlusion-analysis-for-explaining-dnns-d0ad3af9aeb6)

\[26]: [https://www.youtube.com/watch?v=gCJCgQW_LKc](https://www.youtube.com/watch?v=gCJCgQW_LKc)

\[27]: [https://erdem.pl/2022/02/xai-methods-guided-backpropagation](https://erdem.pl/2022/02/xai-methods-guided-backpropagation)

\[28]: [https://medium.com/@chinesh4/generalized-way-of-interpreting-cnns-a7d1b0178709](https://medium.com/@chinesh4/generalized-way-of-interpreting-cnns-a7d1b0178709)

\[29]: [https://leslietj.github.io/2020/07/22/Deep-Learning-Guided-BackPropagation/](https://leslietj.github.io/2020/07/22/Deep-Learning-Guided-BackPropagation/)

\[30]: [https://www.youtube.com/watch?v=BsSmBPmPeYQ](https://www.youtube.com/watch?v=BsSmBPmPeYQ)

\[31]: [https://www.youtube.com/watch?v=DgPaCWJL7XI](https://www.youtube.com/watch?v=DgPaCWJL7XI)

\[32]: [https://www.tensorflow.org/tutorials/generative/deepdream](https://www.tensorflow.org/tutorials/generative/deepdream)

\[33]: [https://www.kaggle.com/code/franciscomesquita/deep-dream-weather-with-efficientnetb3](https://www.kaggle.com/code/franciscomesquita/deep-dream-weather-with-efficientnetb3)

\[34]: [https://analyticsindiamag.com/gradient-ascent-when-to-use-it-in-machine-learning/](https://analyticsindiamag.com/gradient-ascent-when-to-use-it-in-machine-learning/)

\[35]: [https://michaeltang101.medium.com/simple-convolutional-neural-network-with-shap-4fc473472a6d](https://michaeltang101.medium.com/simple-convolutional-neural-network-with-shap-4fc473472a6d)
