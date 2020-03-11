# MACNN
We investigate a novel end-to-end model based on deep learning named as Multi-scale Attention Convolutional Neural Network (MACNN) to solve the time series classification problem. We first apply the multi-scale convolution to capture different scales of information along the time axis by generating different scales of feature maps. Then an attention mechanism is proposed to enhance useful feature maps and suppress less useful ones by learning the importance of each feature map automatically.

## Before Start
To validate the effectiveness of MACNN, we propose two more models for comparison. One is the Single-scale Attention Convolutional Neural Network (SACNN) which has the same architecture with MACNN except for the multi convolutional layers. The other is the Multi-scale Convolutional Neural Network (MCNN) which differs from the architecture of MACNN by removing the attention block.

We test our approach on 85 datasets from the UCR time series classification archive [[www.timeseriesclassification.com]](http://www.timeseriesclassification.com/). Since the label formats of different datasets is not uniformed, we rectify them by starting from 1.

We select the following metrics to evaluate the performance of each method: Wins, Arithmetic Mean Ranking (AMR), Geometric Mean Ranking (GMR), and Mean Error (ME).

## Classification Results
|                                |       |        |       |       |         |          |         |       | 
|--------------------------------|-------|--------|-------|-------|---------|----------|---------|-------|  
|**Dataset**   | **BOSS**    | **COTE**   | **PROP** | **FCN**  | **ResNet**  | **SACNN** | **MCNN** | **MACNN**|
|Adiac|	0.251|	0.19|	0.335|	0.143|	0.174|	0.174|	0.151|	0.151|
|ArrowHead|	0.125|	0.123|	0.14|	0.12|	0.183|	0.137	|0.109|	0.109|
|Beef|	0.385|	0.236|	0.468|	0.25|	0.233|	0.067|	0.067|	0.067|
|BeetleFly|	0.052	|0.079|	0.178	|0.05|	0.2|	0|	0	|0|
|BirdChicken|	0.016|	0.059|	0.152|	0.05|	0.1|	0|	0|	0|
|Car|	0.145|	0.101|	0.201|  0.083|	0.067|	0.1|	0.083|	0.067|
|CBF| 0.002|	0.002|  0.007|	0|	0.006|	0.004|	0|	0|
|ChlorineConcentration|	0.34|	0.264	|0.341|	0.157|	0.172|	0.144	|0.107	|0.107|
|CinCECGTorso	|0.1	|0.017|	0.054	|0.187|	0.229	|0.158	|0.146|	0.101|
|Coffee|  0.011|	0	|0.011|	0	|0|	0	|0|	0|
|Computers|	0.198	|0.23|	0.268|	0.152|	0.176|	0.167|	0.176|	0.156|
|CricketX|	0.236|	0.186	|0.199|	0.185|	0.179	|0.228	|0.136|	0.133|
|CricketY|  0.251|	0.185	|0.206|	0.208|	0.195	|0.203|	0.133	|0.123|
|CricketZ|	0.224	|0.173|	0.196|	0.187|	0.187|	0.2|	0.121	|0.118|
|DiatomSizeReduction|	0.061	|0.075|	0.054|	0.07|	0.069	|0.016|	0.016	|0.016|
|DistalPhalanxOutlineAgeGroup|	0.186|	0.179|	0.232|	0.165|	0.202|	0.237	|0.216|	0.216|
|DistalPhalanxOutlineCorrect|	0.185	|0.195|	0.232|	0.188|	0.18|	0.174	|0.214|	0.21|
|DistalPhalanxTW|	0.327|	0.307|	0.346|	0.21|	0.26	|0.295|	0.281	|0.28|
|Earthquakes|	0.254|	0.253	|0.265|	0.199|	0.214|	0.23|	0.23|	0.223|
|ECG200	|0.11|	0.127|	0.119|	0.1|	0.13|	0.09	|0.06|	0.06|
|ECG5000	|0.06	|0.054|	0.061|	0.059|	0.069|	0.056|	0.051|	0.051|
|ECGFiveDays|	0.017|	0.014	|0.153|	0.015	|0.045	|0.118|	0	|0|
|ElectricDevices|	0.201|	0.117	|0.169	|0.277	|0.272|	0.293|	0.32|	0.262|
|FaceAll|	0.026	|0.01	|0.024|	0.071|	0.166|	0.036|	0.078|	0.072|
|FaceFour|	0.004|	0.15	|0.121|	0.068|	0.068	|0.068|	0.034|	0.023|
|FacesUCR	|0.049|	0.033|	0.052|	0.052|	0.042	|0.041|	0.017|	0.016|
|FiftyWords|	0.298|	0.199	|0.179|	0.321|	0.273|	0.244|	0.136	|0.123|
|Fish|	0.031|	0.038|	0.087	|0.029|	0.011	|0.017|	0.006|	0.006|
|FordA|	0.081	|0.045|	0.249|	0.094|	0.072|	0.06	|0.04	|0.04|
|FordB|	0.089|	0.071|	0.243	|0.117|	0.1|	0.179|	0.136|	0.122|
|GunPoint|	0.006	|0.008|	0.026|	0	|0.007|	0|	0|	0|
|Ham|	0.164	|0.195|	0.237|	0.238|	0.219	|0.257	|0.171|	0.171|
|HandOutlines|	0.097|	0.106	|0.12	|0.224|	0.139|	0.065|	0.049|	0.049|
|Haptics|	0.541	|0.483|	0.549	|0.449	|0.494|	0.523|	0.442|	0.438|
|Herring|	0.395|	0.368	|0.434|	0.297|	0.406	|0.266|	0.313|	0.297|
|InlineSkate|	0.497|	0.474|	0.524	|0.589|	0.635|	0.515|	0.504|	0.5|
|InsectWingbeatSound|	0.49|	0.361|	0.419|	0.598|	0.469	|0.526|	0.347	|0.347|
|ItalyPowerDemand|	0.134	|0.03|	0.049|	0.03|	0.04	|0.028	|0.028|	0.028|
|LargeKitchenAppliances|	0.163|	0.1|	0.184|	0.104|	0.107|	0.083|	0.077|0.075|
|Lightning2	|0.19	|0.215|	0.165	|0.197|	0.246|	0.213|	0.164|	0.164|
|Lightning7|	0.334|	0.201|	0.237|	0.137	|0.164|	0.164|	0.137|	0.123|
|Mallat|	0.051	|0.026|	0.039	|0.02|	0.021|	0.025|	0.018|	0.016|
|Meat	|0.02|	0.019|	0.022|	0.033|	0|	0.017	|0.017|	0|
|MedicalImages|	0.285|	0.215|	0.24|	0.208	|0.228|	0.22|	0.221|	0.203|
|MiddlePhalanxOutlineAgeGroup|	0.334|	0.278|	0.391	|0.232|	0.24	|0.39	|0.383	|0.367|
|MiddlePhalanxOutlineCorrect|	0.192|	0.199|	0.218|	0.205|	0.207|	0.155|	0.165|	0.155|
|MiddlePhalanxTW	|0.463|	0.413|	0.475|	0.388	|0.393	|0.409|	0.435	|0.396|
|MoteStrain	|0.154	|0.098|	0.125|	0.05|	0.105	|0.082|	0.082|	0.082|
|NonInvasiveFetalECGThorax1	|0.159	|0.071|	0.151	|0.039|	0.052|	0.051|	0.043|	0.043|
|NonInvasiveFetalECGThorax2	|0.096|	0.054	|0.086	|0.045|	0.049	|0.047|	0.038|	0.038|
|OliveOil|	0.13|	0.099	|0.121|	0.167	|0.133	|0.133|	0.099	|0.099|
|OSULeaf|	0.033	|0.051|	0.188|	0.012|	0.021|	0.017|	0.017|	0.012|
|PhalangesOutlinesCorrect	|0.179	|0.217|	0.22|	0.174|	0.175|	0.146|	0.163	|0.162|
|Phoneme	|0.744|	0.638	|0.701|	0.655|	0.676|	0.653	|0.669|	0.651|
|Plane|	0.002	|0	|0	|0|	0	|0	|0|	0|
|ProximalPhalanxOutlineAgeGroup	|0.181|	0.152|	0.195	|0.151|	0.151	|0.127|	0.132	|0.127|
|ProximalPhalanxOutlineCorrect	|0.133	|0.129|	0.161|	0.1	|0.082|	0.069	|0.065|	0.065|
|ProximalPhalanxTW	|0.227	|0.185|	0.241	|0.19|	0.193|	0.2|	0.2	|0.19|
|RefrigerationDevices|	0.215|	0.258	|0.324|	0.467|	0.472|	0.395|	0.4|	0.387|
|ScreenType|	0.414	|0.349	|0.446|	0.333	|0.293|	0.355	|0.368|	0.336|
|ShapeletSim	|0|	0.036	|0.173	|0.133|	0	|0.011	|0	|0|
|ShapesAll|	0.091	|0.089	|0.114	|0.102	|0.088|	0.083|	0.063|	0.057|
|SmallKitchenAppliances	|0.25|	0.212|	0.297	|0.197|	0.203	|0.192|	0.184|	0.179|
|SonyAIBORobotSurface1|	0.103	|0.101| 0.206	|0.032|	0.015	|0.017|	0.013|	0.013|
|SonyAIBORobotSurface2|	0.112	|0.04	|0.13|	0.038|	0.038|	0.012|	0.032|	0.032|
|StarLightCurves|	0.022	|0.02	|0.059|	0.033	|0.025|	0.022	|0.02|	0.02|
|Strawberry	|0.03	|0.037	|0.041	|0.031|	0.042	|0.019|	0.022	|0.019|
|SwedishLeaf|	0.082	|0.033	|0.084	|0.034|	0.042|	0.024	|0.024	|0.024|
|Symbols	|0.039	|0.047	|0.043|	0.038|	0.128|	0.014|	0.012|	0.009|
|SyntheticControl	|0.032|	0.001	|0.006	|0.01	|0	|0|	0	|0|
|ToeSegmentation1	|0.071	|0.066|	0.212|	0.031|	0.035|	0.009	|0.026|	0.026|
|ToeSegmentation2	|0.04	|0.049|	0.093	|0.085|	0.138|	0.046	|0.054|	0.046|
|Trace	|0	|0|	0.004	|0	|0	|0	|0	|0|
|TwoLeadECG|	0.016|	0.018	|0.042|	0	|0	|0.001|	0	|0|
|TwoPatterns	|0.009	|0|	0	|0.103|	0	|0.016|	0	|0|
|UWaveAll|	0.056	|0.035	|0.032	|0.174|	0.132|	0.125|	0.053|	0.038|
|UWaveX	|0.247	|0.17	|0.195	|0.246|	0.213|	0.2|	0.159	|0.153|
|UWaveY	|0.339	|0.234|	0.27	|0.275|	0.332|	0.295|	0.223|	0.211|
|UWaveZ	|0.305|	0.241|	0.274	|0.271|	0.245|	0.235|	0.209|	0.205|
|Wafer|	0.001	|0.001|	0.003|	0.003|	0.003|	0	|0	|0|
|Wine|	0.088|	0.097|	0.113	|0.111|	0.204|	0.111|	0.13|	0.111|
|WordSynonyms|	0.341	|0.252|	0.222	|0.42|	0.368|	0.389	|0.238|	0.226|
|Worms	|0.265	|0.275	|0.356	|0.331	|0.381	|0.104|	0.13	|0.117|
|WormsTwoClass|	0.19	|0.215|	0.283	|0.271	|0.265|	0.117|	0.169	|0.104|
|Yoga|	0.09	|0.102	|0.115	|0.155|	0.142|	0.1	|0.08	|0.079|
|Wins|	7	|13|	4|	16|	10	|21	|31|	55|
|AMR	|5.518	|4.341|	6.529|	4.459|	4.965|	3.765	|2.6	|1.706|
|GMR	|4.884|	3.696	|5.961|	3.64	|4.258	|3.056|	2.078|	1.434|
|ME|	0.1666	|0.1421	|0.1882|	0.156	|0.1615|	0.1413|	0.1253|	0.1181|


