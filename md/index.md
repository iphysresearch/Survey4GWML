---
Title:   Gravitational Wave Data Analysis with Machine Learning
Summary: A survey on gravitational waves researches using machine learning techniques.
Authors: He Wang
Date:    July 14, 2020
blank-value:
some_url: https://github.com/iphysresearch/Survey4GWML
---


><h2>Gravitational Wave Data Analysis with Machine Learning</h2>

>This page will give an overview of some problems in gravitational wave data analysis and how researchers are trying to solve them with machine learning. It will include improving data quality, searches for binary black holes and unmodelled gravitational wave bursts, and the astrophysics of gravitational wave sources. I do not include every study in these areas but will do my best.

---

# 1. Conferences & Workshops

- (Dec 8, 2017) - [Deep Learning for Physical Sciences](https://dl4physicalsciences.github.io) (workshop at NeurIPS)
- (Oct 17, 2018) - [Deep Learning for Multimessenger Astrophysics: Real-time Discovery at Scale](http://www.ncsa.illinois.edu/Conferences/DeepLearningLSST/)
- (Sep 10, 2019) - [Fast Machine Learning](https://indico.cern.ch/event/822126/) Workshop at Fermi National Accelerator Laboratory
- (Dec 14, 2019) - [Machine Learning and the Physical Sciences](https://ml4physicalsciences.github.io) (workshop at NeurIPS)
- (March 9, 2020) - [CA17137](https://github.com/zerafachris/g2net_2nd_training_school_malta_mar_2020) - A network for Gravitational Waves, Geophysics and Machine Learning - 2nd Training School ([G2NET](https://www.g2net.eu))
- (April 21, 2020) - Machine Learning for Physicists [2020](https://pad.gwdg.de/s/HJtiTE__U)
- (6-8 July 2020) - Ellis Fellows Program Quantum and Physics based Machine Learning (QPhML) ([2020](https://ellisqphml.github.io/qphml2020), [2019](http://dalimeeting.org/dali2019b/workshop-05-04.html))
- (24 Jul, 2020) - Mathematics, Physics & Machine Learning - $(M \cup \Phi) \cap M L$ -  ([Home](https://mpml.tecnico.ulisboa.pt))
- (19 Oct, 2020) - [2020 Accelerated Artificial Intelligence for Big-Data Experiments Conference](http://www.ncsa.illinois.edu/Conferences/AcceleratedAINCSA/)
- (Sep 9, 2020) - [Advances in Computational Relativity](https://icerm.brown.edu/programs/sp-f20)
- (Nov 16, 2020) - [Statistical Methods for the Detection, Classification, and Inference of Relativistic Objects](https://icerm.brown.edu/programs/sp-f20/w4/)


---


# 2. General Reports & Reviews

>Modern deep learning methods have entered the field of physics which can be tasked with **learning physics from raw data when no good mathematical models are available**. They are also part of mathematical model and machine learning hybrids, formed to reduce computational costs by having the mathematical model train a machine learning model to perform its job, or to improve the fit with observations in settings where the mathematical model can’t incorporate all details (think noise).

- **[Coughlin (2020) [@coughlin2020lessons] (Nature Astronomy)]** - Lessons from Counterpart Searches in Ligo and Virgo’s Third Observing Campaign
- **[Zdeborová (2020) [@zdeborova2020understanding] (Nature Physics)]** - Understanding Deep Learning Is Also a Job for Physicists
- **[Cuoco et al. (2020) [@cuoco2020enhancing] (2005.03745)]** - Enhancing Gravitational-wave Science with Machine Learning
- **[Huerta et al. (2020) [@huerta2020convergence] (2003.08394)]** - Convergence of Artificial Intelligence and High Performance Computing on Nsf-supported Cyberinfrastructure
- **[Fluke et al. (2020) [@fluke2020surveying] (WIRDMKD)]** - Surveying the reach and maturity of machine learning and artificial intelligence in astronomy
- [Viewpoint: Machine Learning Tackles Spacetime](https://physics.aps.org/articles/v13/40) By [Enrico Rinaldi](https://physics.aps.org/authors/enrico_rinaldi)  March 23, 2020 - Physics 13, 40
- **[Huerta et al. (2019) [@huerta2019supporting] (CSBS)]** - Supporting High-Performance and High-Throughput Computing for Experimental Science
-  **[Huerta et al. (2019) [@huerta2019enabling] (Nature Reviews Physics)]** - Enabling Real-Time Multi-Messenger Astrophysics Discoveries with Deep Learning
- **[Allen et al. (2019) [@allen2019deep] (1902.00522)]** - Deep Learning for Multi-Messenger Astrophysics: A Gateway for Discovery in the Big Data Era
- **[Eadie et al. (2019) [@eadie2019realizing] (1909.11714)]** - Realizing the Potential of Astrostatistics and Astroinformatics
- **[Foley et al. (2019) [@foley2019gravity] (1903.04553)]** - Gravity and Light-Combining Gravitational Wave and Electromagnetic Observations in the 2020s


---

# 3. Improving Data Quality

Machine learning techniques have proved to be powerful tools in analyzing complex problems by learning from large example datasets. They have been applied in GW science from as early as **[Lightman et al. (2006) [@lightman2006prospects] (JPCS)]** to the study of glitches  **[Essick et al. (2013) [@essick2013optimizing] (CQG); Biswas et al. (2013) [@biswas2013application] (PRD)]** and other problems, such as signal characterization  **[Baker et al. (2015) [@baker2015multivariate] (PRD)]** . For example, Gstlal-iDQ **[Vaulin et al. (2013) [@2013RuslanVauliniDQRealTime]]** (a streaming machine learning pipeline based on **[Essick et al. (2013) [@essick2013optimizing] (CQG)]** and **[Biswas et al. (2013) [@biswas2013application] (PRD)]** reported the probability that there was a glitch in $h(t)$ based on the presence of glitches in witness sensors at the time of the event. In O2, iDQ was used to vet unmodeled low-latency pipeline triggers automatically. 


## Glitch Classification

Some glitches occur only in the GW data channel. We can try and eliminate them by classifying them into different types to help identify their origin. Unfortunately, there is a number of identified classes of glitches for which mitigation methods are not yet understood. For these glitch classes, understanding how searches can separate instrumental transients from similar astrophysical signals is the highest priority **[Davis et al. (2020) [@davis2020utilizing] (CQG)]**.

- PCA based
    - Early ML studies for glitch classification used Principal Component Analysis (PCA) and Gaussian Mixture Models (GMM). (See **[Powell et al. (2015) [@powell2015classification] (CQG)]** test on simulated data & **[Powell et al. (2017) [@powell2017classification] (CQG)]** test on real data). A trigger generator finds the glitches. The time series of whitened glitches are stored in a matrix D on which PCA is performed. See more on **[Powell (2017) [@powell2017model] (PhD Thesis); Cuoco (2018) [@cuoco2018strategy] (Workshop)]**
    
    - PCA is an orthogonal linear transformation that transforms a set of correlated variables into another set of linearly uncorrelated variables, called Principal Components (PCs).  The matrix $D$ is factored so that $D= U\Sigma V^T$ where $V=A^TA$, $\Sigma$ contains eigenvalues, and $U$ is the PCs. PC coefficients are calculated by taking the dot product of the PCs and the whitened glitch. Then GMM clustering is applied to the PC coefficients. These studies were then improved with the use of Neural Networks.
>

- CNN (Images feature)

    - **[Razzano & Cuoco 2018 [@razzano2018image] (CQG)]** apply a CNN to simulated glitches. They build images that cover 2 seconds around each glitch from the whitened time series. Simulated six families of signals. Training, validation, and test set with ratio 70:15:15.  Accuracy in the order of ≈98-99% on multiclass classification 
    - **[George et al. (2018) [@george2018classification] (PRD)]**: “Deep learning techniques are a promising tool for the recognition and classification of glitches. We present a classification pipeline that exploits Convolutional Neural Networks to classify glitches starting from their time frequency evolution represented as images. We evaluated the classification accuracy on simulated glitches, showing that the proposed algorithm can automatically classify glitches on very fast timescales and with high accuracy, thus providing a promising tool for online detector characterization.” 
>

- Wavelet-based (Time series feature)

    - **[Cuoco et al. (2018) [@cuoco2018wavelet] (IEEE)]** - Wavelet-Based Classification of Transient Signals for Gravitational Wave Detectors
>

- GravitySpy Project

    * GravitySpy **[Zevin et al. (2017) [@zevin2017gravity] (CQG); Coughlin et al. (2019) [@coughlin2019classifying] (PRD)]** uses citizen scientists to produce training sets for machine learning glitch classification.
    * (How do I try it myself?) Log into [gravityspy.org](https://www.gravityspy.org) to try classifying glitches. Download already labelled LIGO glitches for training your algorithm from zenodo [One](https://zenodo.org/record/1476156) / [Two](https://zenodo.org/record/1476551).
>



- Others / Pending:
     * **[Staats & Cavaglià (2018) [@cavaglia2018finding] (1812.05225)]** - Finding the origin of noise transients in LIGO data with machine learning (**Karoo GP**)
     * **[Mukund et al. (2017) [@mukund2017transient] (PRD)]** - Transient classification in LIGO data using difference boosting neural network (**Wavelet-DBNN, India**)
     * **[Llorens-Monteagudo et al. (2019) [@llorens2019classification] (CQG)]** - Classification of gravitational-wave glitches via dictionary learning (**Dictionary learning**)
     * Low latency transient detection and classification (I. Pinto, V. Pierro, L. Troiano, E. Mejuto-Villa, V. Matta, P. Addesso)
     * **[George et al. (2017) [@george2017deep] (1706.07446)]** - Deep Transfer Learning: A new deep learning glitch classification method for advanced LIGO (**Deep Transfer Learning**)
     * **[George et al. (2018) [@george2018classification] (PRD)]** - Classification and unsupervised clustering of LIGO data with Deep Transfer Learning (**Deep Transfer Learning**)
     * **[Astone et al. (2018) [@astone2018new] (PRD)]** - New method to observe gravitational waves emitted by core collapse supernovae (**RGB image SN CNN**)
     * **[Colgan et al. (2020) [@colgan2020efficient] (PRD)]** - Efficient gravitational-wave glitch identification from environmental data through machine learning
     * **[Bahaadini et al. (2017) [@bahaadini2017deep] (IEEE)]** - Deep multi-view models for glitch classification
     * **[Bahaadini et al. (2018) [@bahaadini2018machine] (Info. Sci.)]** - Machine learning for Gravity Spy: Glitch classification and dataset
     * **[Bahaadini et al. (2018) [@bahaadini2018direct] (IEEE)]** - Direct: Deep Discriminative Embedding for Clustering of Ligo Data
     * Young-Min Kim - Noise Identification in Gravitational wave search using Artificial Neural Networks ([PDF](https://gwdoc.icrr.u-tokyo.ac.jp/DocDB/0017/G1301718/003/KJ-KAGRA_20130610_YMKIM.pdf))  (4th K-J workshop on KAGRA @ Osaka Univ.)



## Glitch cancellation / GW denosing

>

- Pending:
    - **[Cuoco et al. (2001) [@cuoco2001line] (CQG)]** - On-line power spectra identification and whitening for the noise in interferometric gravitational wave detectors
    - **[Torres-Forné (2016) [@torres2016denoising] (PRD)]** - Denoising of Gravitational Wave Signals Via Dictionary Learning Algorithms
    - **[Torres-Forné (2018) [@torres2018total] (PRD)]** - Total-variation methods for gravitational-wave denoising: Performance tests on Advanced LIGO data
    - **[Torres-Forné (2020) [@torres2020application] (PRD)]** - Application of dictionary learning to denoise LIGO's blip noise transients
    - **[Shen et al. (2017) [@Shen2017jkj] (1711.09919)]** - Denoising Gravitational Waves using Deep Learning with Recurrent Denoising Autoencoders
    - **[Shen et al. (2019) [@Shen2019ohi] (IEEE)]** - Denoising Gravitational Waves with Enhanced Deep Recurrent Denoising Auto-encoders
    - **[Wei & Huerta (2020) [@WEI2020135081] (PLB)]** - Gravitational wave denoising of binary black hole mergers with deep learning
    - **[Vajente et al. (2020) [@Vajente2019ycy] (PRD)]** - Machine-learning nonstationary noise out of gravitational-wave detectors
    - **[Alimohammadi et al. (2020) [@Alimohammadi2020wtj] (2005.11352)]** - A Data-Driven Approach for Extraction of Event-Waveform: Application to Gravitational Waves
    - **[Ormiston et al. (2020) [@ormiston2020noise] (PRR)]** - Noise reduction in gravitational-wave data via deep learning
    - **[Essick et al. (2020) [@Essick2020qpo] (2005.12761)]** - iDQ: Statistical Inference of Non-gaussian Noise with Auxiliary Degrees of Freedom in Gravitational-wave Detectors


---

# 4. Compact Binary Coalesces (CBC)

## Waveform Modelling

Signal models are needed for matched filtering and parameter estimation. Solutions of the Einstein equations can be obtained with numerical relativity simulations - High computational cost! LIGO and Virgo rely on approximate solutions obtained through phenomenological modelling. Gaussian process regression has been used to produce new waveforms by providing a direct interpolation between numerical simulations. For Example, **[Docter et al. (2017) [@Doctor2017csx] (PRD)]**

- Machine Learning for waveform generation: Use optimal waveform generators, i.e., machine learning models trained with numerical relativity waveforms **[Blackman et al. (2017) [@Blackman2017dfb] (PRD); Huerta et al. (2018) [@Huerta2017kez] (PRD)]** (For eccentric black hole mergers **[Varma et al. (2019) [@Varma2018mmi] (PRD); Varma et al. (2019) [@Varma2019csw] (PRR)]**)
>

- Pending:
    - **[Rebei et al. (2019) [@Rebei2018lzh] (PRD)]** - Fusing numerical relativity and deep learning to detect higher-order multipole waveforms from eccentric binary black hole mergers
    - **[Chua (2017) [@Chua2017xyi] (PhD Thesis)]** - Topics in gravitational-wave astronomy: Theoretical studies, source modelling and statistical methods
    - **[Chua et al. (2019) [@Chua2018woh] (PRL)]** - Reduced-oruder modeling with artificial neurons for gravitational-wave inference
    - **[Setyawati et al. (2020) [@Setyawati2019xzw] (CQG)]** - Regression Methods in Waveform Modeling: A Comparative Study
    - **[Tiglio & Villanueva (2019) [@Tiglio2019yfe] (1911.00644)]** - On ab initio closed-form expressions for gravitational waves
    - **[Rosofsky & Huerta (2020) [@Rosofsky2020zsl] (PRD)]** - Artificial Neural Network Subgrid Models of 2D Compressible Magnetohydrodynamic Turbulence
    - **[Schmidt (2019) [@2019SchmidtGWModelling] (Masters Thesis)]** - Gravitational Wave Modelling with Machine Lerning
    - **[Chen et al. (2020) [@Chen2020lzc] (2008.03313)]** - Observation of Eccentric Binary Black Hole Mergers with Second and Third Generation Gravitational Wave Detector Networks



## Signal Detection

>

- Matched filter searches:

    * Searches for gravitational wave signals from compact binaries use matched filtering. GW detector noise is non-Stationary and non-Gaussian.
    * Discrete template bank is built to cover the mass-spin parameter space for potential sources. Density of the bank is determined by the minimum overlap requirement of 0.97 so only 3% of SNR is lost. Assume spins are aligned to the orbital angular momentum. Do not account for tidal deformability of neutron stars.
    * Existing matched filtering searches are close to optimal and quite fast (~1 min latency): Matched filtering is very close to optimal sensitivity; Not all of the parameter space is covered; Non-Gaussian noise is well understood;
    * ML searches can help if they are fast and do as well on detector noise artefacts.
>

- Machine learning CBC searches:

    Binary black holes are easy but binary neutron stars are hard - they are longer in duration and broader in bandwidth; Spin (with precession) and eccentricity expand the parameter space; Point estimate parameter estimation is useless - sorry to be harsh; The background estimation problem - a long standing issue that people feel quite strongly about;  

    - **[George & Huerta (2018) [@George2016hay] (PRD)]** use a system of two deep convolutional neural networks to rapidly detect CBC signals. They use time series as input so that they can find signals too small for image recognition. They find their method significantly outperforms conventional machine learning techniques, achieves similar performance compared to matched-filtering while being several orders of magnitude faster.
        - Deep learning for real-time classification and regression of gravitational waves in simulated LIGO noise. **[George & Huerta (2018) [@George2016hay] (PRD)]**
        - Deep learning for real-time classification and regression of gravitational waves in real advanced LIGO noise. **[George & Huerta (2018) [@George2017pmj] (PLB); George & Huerta (2018) [@George2017vlv] (NiPS Summer School)]**
        - Deep learning at scale for real-time gravitational wave parameter estimation and tests of general relativity **[Shen et al. (2019) [@Shen2019vep] (1903.01998)].** First Bayesian Neural Network model at scale to characterize a 4-D signal manifold with 10M+ templates. Trained with over ten million waveforms using 1024 nodes (64 processors/node) on an HPC platform optimized for deep learning researches (Theta at Argonne National Lab). Inference time is 2 milliseconds for each gravitational wave detection using a single GPU.
    - **[Gabbard et al. (2018) [@Gabbard2017lja] (PRL)]** also perform a CBC search with a basic and standard CNN network to learn to classify between noise and signal+noise classes. They use whitened time series of measured gravitational-wave strain as an input. Train and test on simulated binary black hole signals in synthetic Gaussian LIGO noise. They find they can reproduce the sensitivity of a matched filter search. i.e. the CNN approach can achieve the same sensitivity as a matched filtering analysis. Classification of 2-D BBH signals in simulated LIGO noise. 
    - **[Li et al. (2017) [@li2017method] (1712.00356)]** - A Method Of Detecting Gravitational Wave Based On Time-frequency Analysis And Convolutional Neural Networks
    - **[Kapadia et al. (2017) [@kapadia2017classifier] (PRD)]** - Classifier for gravitational-wave inspiral signals in nonideal single-detector data
    - **[Cao et al. (2018) [@2018CaoInitialstudyapplication] (JHNU)]** - Initial study on the application of deep learning to the Gravitational Wave data analysis
    - **[Fan et al. (2019) [@fan2019applying] (SCI CHINA PHYS MECH)]** - Applying deep neural networks to the detection and space parameter estimation of compact binary coalescence with a network of gravitational wave detectors
    - **[Luo et al. (2019) [@luo2020extraction] (Front. Phys.)]** - Extraction of gravitational wave signals with optimized convolutional neural network
    - **[Lin et al. (2019) [@lin2020binary] (Front. Phys.)]** - Binary neutron stars gravitational wave detection based on wavelet packet analysis and convolutional neural networks
    - **[Wang et al. (2019) [@wang2019identifying] (New J. Phys.)]** - Identifying extra high frequency gravitational waves generated from oscillons with cuspy potentials using deep neural networks
    - **[Krastev (2020) [@krastev2020real] (PLB)]** - Real-time detection of gravitational waves from binary neutron stars using artificial neural networks
    - **[Mytidis et al. (2019) [@mytidis2019sensitivity] (PRD)]** - Sensitivity study using machine learning algorithms on simulated $r$-mode gravitational wave signals from newborn neutron stars
    - **[Gebhard et al. (2017) [@gebhard2017convwave] (Workshop)]** - Convwave: Searching for gravitational waves with fully convolutional neural nets
    - **[Gebhard et al. (2019) [@gebhard2019convolutional] (PRD)]** - onvolutional neural networks: A magic bullet for gravitational-wave detection?
    - **[Bresten & Jung (2019) [@bresten2019detection] (1910.08245)]** - Detection of gravitational waves using topological data analysis and convolutional neural network: An improved approach
    - **[Nakano et al. (2019) [@nakano2019comparison] (PRD)]** - Comparison of various methods to extract ringdown frequency from gravitational wave data
    - **[Santos et al. (2020) [@santos2020gravitational] (2003.09995)]** - Gravitational Wave Detection and Information Extraction via Neural Networks
    - **[Corizzo et al. (2020) [@corizzo2020scalable] (EXPERT SYST APPL)]** - Scalable auto-encoders for gravitational waves detection from time series data
    - **[Li et al. (2020) [@li2020machine] (2003.13928)]** - Machine Learning for the Gravitational Wave Observation with Pulsar Timing Array
    - **[Marulanda et al. (2020) [@marulanda2020deep] (2004.01050)]** - Deep learning Gravitational Wave Detection in the Frequency Domain
    - **[Wang et al. (2020) [@wang2020gravitational] (PRD)]** - Gravitational-wave signal recognition of LIGO data by deep learning
    - **[Sadeh (2020) [@sadeh2020data] (ApJ)]** - Data-driven Detection of Multimessenger Transients
    - **[Kim et al. (2020) [@kim2020ranking] (PRD)]** - Ranking candidate signals with machine learning in low-latency searches for gravitational waves from compact binary mergers
    - **[Li et al. (2020) [@li2020some] (Front. Phys.)]** - Some optimizations on detecting gravitational wave using convolutional neural network
    - **[Schafer (2019) [@schafer2019analysis] (Masters Thesis)]** - Analysis of Gravitational-Wave Signals from Binary Neutron Star Mergers Using Machine Learning
    - **[Schafer et al. (2020) [@schafer2020detection] (2006.01509)]** - Detection of gravitational-wave signals from binary neutron star mergers using machine learning
    - **[Lin & Wu (2020) [@Lin2020aps] (2007.04176)]** - Detection of Gravitational Waves Using Bayesian Neural Networks
    - **[Chauhan (2020) [@Chauhan2020wzy] (2007.05889)]** - Deep Learning Model to Make Gravitational Wave Detections from Weak Time-series Data


## Low-latency source-properties (EM-bright)

LIGO & Virgo provide two probabilities in low-latency. **[Chatterjee et al. (2020) [@chatterjee2020machine] (ApJ)]** The probability that there is a neutron star in the CBC system, P(HasNS). The probability that there exists tidally disrupted matter outside the final coalesced object after the merger, P(HasRemnant). Matched filter searches give point estimates of mass and spin but they have large errors! To solve this a machine learning classification is used. (scikit learn K nearest neighbours, also tried random forest). A training set is created by injecting fake signals into gravitational wave data and performing a search. This then produces a map between true values and matched filter search point estimates which is learnt by the classifier.


## Parameter Estimation (PE)

Characterized by 15 parameters. Masses, spins, distance, inclination, sky position, polarization. LIGO & Virgo GWTC-1 **[Abbott et al. (2019) [@abbott2019gwtc] (PRX)]** . Bayes' Theorem ([LAAC tutorial 2019 - Virginia d'Emilio](https://git.ligo.org/virginia.demilio/pe-tutorial-laac19))
>

* MCMC and Nested Sampling
  
    * We have two main PE codes LALInference and Bilby. MCMC Random steps are taken in parameter space, according to a proposal distribution, and accepted or rejected according to the Metropolis-Hastings algorithm. Nested sampling can also compute evidences for model selection. **[Skilling (2006) [@skilling2006nested] (Bayesian Anal.)]**
>

* Machine Learning Parameter Estimation

    * **The current “holy grail” of machine learning for GWs.**
    * BAMBI: blind accelerated multimodal Bayesian inference combines the benefits of nested sampling and artificial neural networks. **[Graff et al. (2012) [@graff2012bambi] (Mon. Not. Roy. Astron. Soc.)]** An artificial neural network learns the likelihood function to increase significantly the speed of the analysis. @2015Veitch-Parameterestimationcompact
    * Chua et al. **[Chua & Vallisneri (2020) [@chua2020learning] (PRL)]** produce Bayesian posteriors using neural networks.
    * Gabbard et al. **[Gabbard et al. (2019) [@gabbard2019bayesian] (1909.06296)]** use a conditional variational autoencoder pre-trained on binary black hole signals. We use a variation inference approach to produce samples from the posterior. It does NOT need to be trained on precomputed posteriors. It is ~6 orders of magnitude faster than existing sampling techniques. For Chris Messenger, it seems completely obvious that all data analysis will be ML in 5-10 years.
    - **[Chatterjee et al. (2020) [@chatterjee2020machine] (ApJ)]** - A Machine Learning-based Source Property Inference for Compact Binary Mergers
    - **[Fan et al. (2019) [@fan2019applying] (SCI CHINA PHYS MECH)]** - Applying deep neural networks to the detection and space parameter estimation of compact binary coalescence with a network of gravitational wave detectors
    - **[Green et al. (2020) [@green2020gravitational] (2002.07656)]** - Gravitational-wave parameter estimation with autoregressive neural network flows
    - **[Carrillo et al. (2016) [@carrillo2016parameter] (GRG)]** - Parameter estimates in binary black hole collisions using neural networks
    - **[Carrillo et al. (2018) [@carrillo2018one] (INT J MOD PHYS D)]** - One parameter binary black hole inverse problem using a sparse training set
    - **[Chatterjee et al. (2019) [@chatterjee2019using] (PRD)]** - Using deep learning to localize gravitational wave sources
    - **[Yamamoto & Tanaka (2020) [@yamamoto2020use] (2002.12095)]** - Use of conditional variational auto encoder to analyze ringdown gravitational waves
    - **[Haegel & Husa (2020) [@haegel2020predicting] (CQG)]** - Predicting the properties of black-hole merger remnants with deep neural networks
    - **[Varma et al. (2019) [@varma2019high] (PRL)]** - High-Accuracy Mass, Spin, and Recoil Predictions of Generic Black-Hole Merger Remnants
    - **[Belgacem et al. (2020) [@belgacem2020gaussian] (PRD)]** - Gaussian processes reconstruction of modified gravitational wave propagation
    - **[Li et al. (2020) [@li2020machine] (2003.13928)]** - Machine Learning for the Gravitational Wave Observation with Pulsar Timing Array
    - **[Khan et al. (2020) [@khan2020physics] (2004.09524)]** - Physics-inspired deep learning to characterize the signal manifold of quasi-circular, spinning, non-precessing binary black hole mergers
    - **[Rover et al. (2009) [@rover2009bayesian] (PRD)]** - Bayesian reconstruction of gravitational wave burst signals from simulations of rotating stellar core collapse and bounce
    - **[Engels et al. (2014) [@engels2014multivariate] (PRD)]** - Multivariate regression analysis of gravitational waves from rotating core collapse
    - **[Wong et al. (2020) [@Wong2020wvd] (2007.10350)]** - Gravitational-wave signal-to-noise interpolation via neural networks
    - **[Green & Gair (2020) [@Green2020dnx] (2008.03312)]** - Complete Parameter Inference for Gw150914 Using Deep Learning
    - **[Glüsenkamp (2020)[@Glusenkamp2020gtr] (2008.05825)]** - Unifying Supervised Learning and VAEs -- Automating Statistical Inference in High-energy Physics


## Population Studies

- **[Vinciguerra et al. (2017) [@vinciguerra2017enhancing] (CQG)]** - Enhancing the significance of gravitational wave bursts through signal classification
- Now that we have started to detect a population of black hole signals we can try to do population studies to try and understand signals formation mechanisms. Population properties paper from O1+02 **[Abbott et al. (2019) [@abbott2019binary] (ApJ)]** . Uses phenomenological models (like power laws) combined with Bayesian hierarchical modelling. Bayesian hierarchical modelling involves some assumptions of populations mass and spin distributions.  Does not scale well for high dimensional models and a large number of GW detections. 
* We can use unmodelling clustering! In **[Powell et al. (2019) [@powell2019unmodelled] (Mon. Not. Roy. Astron. Soc.)]** we apply unmodelled clustering to masses and spins. Two of the populations have identical mass distributions and different spin. This is difficult because spin is poorly measured. Determine the number of populations and the number of CBC signals in each population.
* (How do I try it myself?) The Gravitational Wave Open Science Center has the data, parameter estimates, and matched filtering tutorials that you can download. You can get [code](https://git.ligo.org/daniel.wysocki/synthetic-PE-posteriors) to produce synthetic parameter estimates for compact binaries.
* **[Deligiannidis et al. (2019) [@deligiannidis2019case] (ICAI)]** - Case Study: Skymap Data Analysis
* **[Wong & Gerosa (2019) [@wong2019machine] (PRD)]** - Machine-learning interpolation of population-synthesis simulations to interpret gravitational-wave observations: A case study
* **[Wong et al. (2020) [@wong2020gravitational] (PRD)]** - Gravitational-wave population inference with deep flow-based generative network
* **[Fasano et al. (2020) [@fasano2020distinguishing] (PRD)]** - Distinguishing Double Neutron Star from Neutron Star-black Hole Binary Populations with Gravitational Wave Observations

--- 

# 5. Continuous Wave Search

- **[Piccinni et al. (2018) [@piccinni2018new] (CQG)]** - A new data analysis framework for the search of continuous gravitational wave signals
- Existing challenges and signal characteristics: Vast parameter space, The parameter space is incredibly large; Likely very weak signals, The signal is incredibly weak - orders of magnitude lower than the noise amplitude;  Leading to traditional searches optimised at fixed computational cost - Generally slow, The dataset is (quite) large, 1year X 1kHz =~10 GB; In the era of open data the LVC and competitors are keen to analyse the data very quickly; Narrow band, However, since we have been limited until now by computational expense - with ML this could no longer be a limit, and hence sensitivity can really improve
- **[Morawski et al. (2020) [@morawski2020convolutional] (Sci.Technol.)]** - Convolutional Neural Network Classifier for the Output of the Time-domain F-statistic All-sky Search for Continuous Gravitational Waves
- **[Dreissigacker et al. (2019) [@dreissigacker2019deep] (PRD)]** (Modelled searches) Based on the success of CNNs for compact binary searches; The task is significantly more difficult here; Fair comparison with fully coherent searches over a broad parameter space; The ML approach is reasonably competitive for the simplest of the cases studied; For 10^6 sec observations at 1kHz perform significantly worse than matched filtering
- **[Dreissigacker & Prix (2020) [@dreissigacker2020deep] (PRD)]** - Deep-learning continuous gravitational waves: Multiple detectors and realistic noise
- **[Covas & Sintes (2019) [@covas2019new] (PRD)]** - New method to search for continuous gravitational waves from unknown neutron stars in binary systems
- **[Bayley et al. (2019) [@bayley2019generalized] (PRD)]** (Unmodelled searches) A very weakly modelled search for weak psuedo-sinusoidal continuous signals; Uses the Viterbi algorithm to efficiently find the maximum sum of power/ statistic across a time-frequency plane (hence SOAP); Requires no templates and runs on “raw” GW data; Is exceptionally good at finding detector line features (also annoying); Extension work applies a CNN to the output for better signal vs line discrimination;
- **[Miller et al. (2019) [@miller2019effective] (PRD)]** - How effective is machine learning to detect long transient gravitational waves from neutron stars in a real search?
- **[Miller (2019) [@2019MILLERUsingMachineLearning] (PhD Thesis)]** - Using Machine Learning and the Hough Transform to Search for Gravitational Waves Due to R-mode Emission by Isolated Neutron Stars
- **[Schafer (2019) [@schafer2019analysis] (Masters Thesis)]** - Analysis of Gravitational-Wave Signals from Binary Neutron Star Mergers Using Machine Learning
- **[Beheshtipour & Papa (2020) [@Beheshtipour2020zhb] (PRD)]** - Deep Learning for Clustering of Continuous Gravitational Wave Candidates
- **[Middleton et al. (2020) [@Middleton2020skz] (PRD)]** - Search for Gravitational Waves from Five Low Mass X-ray Binaries in the Second Advanced Ligo Observing Run with an Improved Hidden Markov Model
- **[Bayley et al. (2020) [@bayley2020robust] (2007.08207)]** - A Robust Machine Learning Algorithm to Search for Continuous Gravitational Waves
- **[Bayley (2020) [@bayley2020non] (PhD Thesis)]** - Non-parametric and Machine Learning Techniques for Continuous Gravitational Wave Searches
- **[Jones & Sun (2020) [@jones2020search] (2007.08732)]** - Search for Continuous Gravitational Waves from Fomalhaut B in the Second Advanced Ligo Observing Run with a Hidden Markov Model
- **[Suvorova et al. (2016) [@suvorova2016hidden] (PRD)]** - Hidden Markov Model Tracking of Continuous Gravitational Waves from a Neutron Star with Wandering Spin
- **[Suvorova et al. (2017) [@suvorova2017hidden] (PRD)]** - Hidden Markov Model Tracking of Continuous Gravitational Waves from a Binary Neutron Star with Wandering Spin. II. Binary Orbital Phase Tracking
- **[Sun & Melatos (2019)] [@sun2019application] (PRD)** - Application of Hidden Markov Model Tracking to the Search for Long-duration Transient Gravitational Waves from the Remnant of the Binary Neutron Star Merger GW170817
- **[Sun et al. (2018) [@sun2018hidden] (PRD)]** - Hidden Markov Model Tracking of Continuous Gravitational Waves from Young Supernova Remnants


---

# 6. Gravitational Wave Bursts

A burst is a gravitational wave signal where the waveform morphology is partially or completely unknown. The source could be an unknown unknown, a supernova, cosmic string, fast radio burst, compact binaries and others. The main burst search is called coherent Wave Burst (cWB).
>

* Coherent Wave Burst (cWB) [Website](https://gwburst.gitlab.io/)
  
    cWB relies upon the excess coherent power in a network of detectors. The data is transformed into time-frequency domain and the clusters of time-frequency pixels above certain energy threshold are identified for each detector. Time frequency map of the single detectors is then combined using the maximisation of the likelihood over all possible sky locations and the events are then ranked according to this likelihood. We can also inform our un-modelled search about the morphology of the expected signal. cWB produces reconstructions of gravitational wave signals. It can detect CBC signals as well as bursts. 
    - **[Drago et al. (2020) [@Drago2020kic] (2006.12604)]** - Coherent Waveburst, a Pipeline for Unmodeled Gravitational-wave Data Analysis
>

* BayesWave

    BayesWave is another standard burst tool. **[Cornish & Littenberg (2015) [@Cornish2014kda] (CQG)]** Models signals as a variable number of sine-Gaussian wavelets with power coherent across detectors. It produces unmodelled waveform reconstructions and can remove glitches that occur during signals. **[Pankow et al. (2018) [@Pankow2018qpo] (PRD)]**
>

* Supernova Search

    Some burst searches are for targeted sources like supernovae. There is not enough supernova waveforms to match filter search but some supernova waveform features are known. The known features from supernova simulations can be incorporated into supernova searches using machine learning.
    
    - **[Astone et al. (2018) [@Astone2018uge] (PRD)]** enhance the efficiency of cWB using a neural network. The network is trained on phenomenological waveforms that represent the g-mode emission in supernova waveforms. They use cWB to prepare images of the data. They use colours to determine which detectors find the signal. They find their method increases the sensitivity of traditional cWB.
    - **[Iess et al. (2020) [@Iess2020yqj] (Sci.Technol.)]** have a different approach that does not involve cWB. They use a trigger generator called WDF to find excess power in the detector. Then they do a neural network classification to decide if the trigger is a signal or noise. They train directly on supernova waveforms. They use both time series and images of data. They obtain high accuracies with both methods and include glitches.
    - **[Chan et al. (2019) [@Chan2019fuz] (1912.13517)]** also train directly on supernova waveforms.  They use only the time series waveforms from different explosion mechanisms.
    - **[Cavaglia et al. (2020) [@Cavaglia2020qzp] (Sci.Technol.)]** - Improving the background of gravitational-wave searches for core collapse supernovae: a machine learning approach
    - **[Stachie et al. (2020) [@stachie2020using] (MNRAS)]** - Using Machine Learning for Transient Classification in Searches for Gravitational-wave Counterparts
>

* Burst analysis

    Jordan McGinn. Thesis: "Generalised gravitational burst searches with Generative Adversarial Networks". Examined the use of Generative Adversarial Networks to generate and interpret large quantities of time-series data. The method was successful in classifying signal data buried within external noise. 

    Uses Generative Adversarial network (GAN) to learn how to make standard burst waveforms. Generation stage has the possibility to make signals spanning all training classes. Discriminator stage has the potential to be a general transient detection tool

    - **[Kovačević et al. (2019) [#Kovacevic2019wpy] (Mon. Not. Roy. Astron. Soc.)]** - Optimizing neural network techniques in classifying Fermi-LAT gamma-ray sources
    - **[Kim et al. (2015) [@kim2015application] (CQG)]** - Application of artificial neural network to search for gravitational-wave signals associated with short gamma-ray bursts
    - **[Oh et al. (2015) [@oh2014application] (J KOREAN ASTRON SOC)]** - Application of Artificial Neural Networks to Search for Gravitational-Wave Signals Associated with Short Gamma-Ray Bursts
    - **[Gayathri et al. (2020) [@2020GayathriEnhancingsensitivitytransient] (2008.01262)]** - Enhancing the Sensitivity of Transient Gravitational Wave Searches with Gaussian Mixture Models
>

* Single Detector Search
  
    30% of gravitational wave data is collected when only 1 detector is in observing mode. Can’t do time slides to measure the background if there is only 1 detector. 

    * **[Cavaglia et al. (2020) [@Cavaglia2020qzp] (Sci.Technol.)]** use machine learning combined with cWB to perform a single detector search for supernovae. They train a genetic programming algorithm on the output parameters of cWB.
    * (How can I try it myself?) You can download some supernova gravitational wave signals [here](http://www.phys.utk.edu/smc/data.html) . You can get KarooGP [here ](http://kstaats.github.io/karoo_gp/). You can get Coherent WaveBurst [here](https://gwburst.gitlab.io/).

---

# 7. GW Cosmology

* **[Khan et al. (2019) [@Khan2018opv] (PLB)]** From the citizen science revolution using the Sloan Digital Sky Survey… ... to large scale discovery using unlabeled images in the Dark Energy Survey using deep learning. 10k+ raw, unlabeled galaxy images from DES clustered according to morphology using RGB filters; Scalable approach to curate datasets, and to construct large-scale galaxy catalogs; Deep transfer learning combined with distributed training for cosmology; Training is completed within 8 minutes achieving state-of-the-art classification accuracy; 
* Real-time detection and characterization of binary black hole mergers + Classification and regression of galaxies across redshift in DES/LSST-type surveys => Hubble constant measurements with probabilistic neural network models **[Wei et al. (2020) [@Wei2019voi] (Mon. Not. Roy. Astron. Soc.)].** Star cluster classification has been predominantly done by human experts; We have designed neural network models that outperform, for the first time, human performance for star cluster classification; Worldwide collaboration of experts in deep learning, astronomy, software and data.
* **[Alexander et al. (2020) [@Alexander2019puy] (APJ)]** - Deep Learning the Morphology of Dark Matter Substructure
* **[Gupta & Reichardt (2020) [@Gupta2020yvd] (2003.06135)]** - Mass Estimation of Galaxy Clusters with Deep Learning I: Sunyaev-Zel'dovich Effect
* **[Sadr & Farsian (2020) [@Sadr2020rje] (2004.04177)]** - Inpainting via Generative Adversarial Networks for CMB data analysis
* **[Philip et al. (2002) [@Philip2002xe] (Astron. Astrophys.)]** - A difference boosting neural network for automated star-galaxy classification
* **[Philip et al. (2012) [@Philip2012vr] (1211.3607)]** - Classification by Boosting Differences in Input Vectors: An application to datasets from Astronomy
* **[Wang et al (2020) [@Wang2020hmn] (2005.07089)]** - ECoPANN: A Framework for Estimating Cosmological Parameters using Artificial Neural Networks
    - 他们这个是当做回归问题在做，和 Huerta 他们的基本逻辑其实一样。至于6 个目标参数的估计，是通过给定 input 数据在相应参数区间上采样后，直接给出后验样本参数估计的（频率学派~）。并不是对某单个数据给出的参数估计（贝叶斯学派）。
    - 实现的是对应观测数据集的宇宙学参数估计

# 8. GW Dynamics

- **[Tamayo et al. (2020) [@tamayo2020predicting] (PNAS)]** - Predicting the Long-term Stability of Compact Multiplanet Systems
  > Sagan学者[Dan Tamayo](https://twitter.com/astrodantamayo/status/1282866485531222022?s=20)介绍了他们在PNAS上发表的一篇利用机器学习技术预测多行星系统的动力学稳定性。(Informative [comments](http://weibointl.api.weibo.com/share/159645756.html?weibo_id=4526731371732253) from 光头怪博士)


# License

* <a rel="license" href="http://creativecommons.org/licenses/by-nc/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/3.0/">Creative Commons Attribution-NonCommercial 3.0 Unported License</a>.

<script type='text/javascript' id='clustrmaps' src='//cdn.clustrmaps.com/map_v2.js?cl=000000&w=400&t=tt&d=BidHz1QnWdqorra8ky71ErAH78XnoVrg9XU-_YzbvZs&co=ffffff&ct=000000'></script>

\bibliography
