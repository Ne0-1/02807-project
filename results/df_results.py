import pandas as pd


df_results = pd.DataFrame(columns=["Model", "Dataset", "Method", 'F1', 'Accuracy'])    # from low to high conf
num_results = 5


# -- J-KNN --
# SMS
# Methods
df_single_results = pd.DataFrame({'Model': ['J-KNN']*num_results,
                                  'Dataset': ['SMS']*num_results, 
                                  'Method': ['Raw/BOW']*num_results,
                                  'F1': [9.025974025974025094e-01,9.090909090909091717e-01,8.709677419354838745e-01,8.982456140350877360e-01,9.185185185185186230e-01],
                                  'Accuracy': [9.729729729729730270e-01,9.783783783783783772e-01,9.711711711711711326e-01,9.738738738738739187e-01,9.801801801801801606e-01],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['J-KNN']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Raw/Q']*num_results,
                                  'F1': [9.337539432176656939e-01,9.523809523809523281e-01,9.312977099236641187e-01,9.315068493150684414e-01,9.462365591397848830e-01],
                                  'Accuracy': [9.810810810810810523e-01,9.882882882882882969e-01,9.837837837837838384e-01,9.819819819819819440e-01,9.864864864864865135e-01],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['J-KNN']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Prep/BOW']*num_results,
                                  'F1': [9.226006191950464341e-01,9.138576779026217345e-01,9.105058365758755823e-01,9.078498293515357975e-01,9.110320284697508431e-01],
                                  'Accuracy': [9.774774774774774855e-01,9.792792792792792689e-01,9.792792792792792689e-01,9.756756756756757021e-01,9.774774774774774855e-01],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['J-KNN']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Prep/Q']*num_results,
                                  'F1': [9.202453987730060403e-01,8.985507246376812640e-01,8.854961832061069016e-01,9.006622516556290758e-01,9.122807017543860253e-01],
                                  'Accuracy': [9.765765765765765938e-01,9.747747747747748104e-01,9.729729729729730270e-01,9.729729729729730270e-01,9.774774774774774855e-01],
                                  })
df_results = df_results.append(df_single_results)

# EMAILS
# Methods
df_single_results = pd.DataFrame({'Model': ['J-KNN']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Raw/BOW']*num_results,
                                  'F1': [9.644513137557959581e-01,9.235880398671095781e-01,9.609120521172638263e-01,9.679389312977099147e-01,9.672386895475819646e-01],
                                  'Accuracy': [9.791099000908265459e-01,9.582198001816530919e-01,9.782016348773842074e-01,9.809264305177112231e-01,9.809264305177112231e-01],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['J-KNN']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Raw/Q']*num_results,
                                  'F1': [9.683860232945090685e-01,9.590443686006826507e-01,9.687500000000000000e-01,9.755301794453506981e-01,9.724238026124819578e-01],
                                  'Accuracy': [9.827429609445957892e-01,9.782016348773842074e-01,9.818346957311534506e-01,9.863760217983651435e-01,9.827429609445957892e-01],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['J-KNN']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Prep/BOW']*num_results,
                                  'F1': [9.432739059967585327e-01,9.070146818923326970e-01,9.486404833836856731e-01,9.480314960629921961e-01,9.454022988505746961e-01],
                                  'Accuracy': [9.682107175295185941e-01,9.482288828337874786e-01,9.691189827429609327e-01,9.700272479564032713e-01,9.654859218891916894e-01],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['J-KNN']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Prep/Q']*num_results,
                                  'F1': [9.468599033816424981e-01,8.906752411575563633e-01,9.525267993874426686e-01,9.304482225656878214e-01,9.428571428571429491e-01],
                                  'Accuracy': [9.700272479564032713e-01,9.382379654859218654e-01,9.718437783832879484e-01,9.591280653950953194e-01,9.636693914623070123e-01],
                                  })
df_results = df_results.append(df_single_results)


# -- TF-IDF-KNN --

# SMS
# Methods
df_single_results = pd.DataFrame({'Model': ['TF-IDF-KNN']*num_results, 
                                  'Dataset': ['SMS']*num_results, 
                                  'Method': ['Raw/BOW']*num_results,
                                  'F1': [0.8754448398576512, 0.864406779661017, 0.8783783783783783, 0.8340425531914893, 0.8057553956834532],
                                  'Accuracy': [0.9684968496849685, 0.9711711711711711, 0.9675675675675676, 0.9648648648648649, 0.9513513513513514],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['TF-IDF-KNN']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Raw/Q']*num_results,
                                  'F1': [0.9166666666666666, 0.9040000000000001, 0.9320388349514562, 0.8784313725490196, 0.8754208754208754],
                                  'Accuracy': [0.9783978397839784, 0.9783783783783784, 0.981081081081081, 0.972072072072072, 0.9666666666666667],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['TF-IDF-KNN']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Prep/BOW']*num_results,
                                  'F1': [0.8692579505300354, 0.8451882845188285, 0.8737201365187715, 0.8547717842323652, 0.8327402135231318],
                                  'Accuracy': [0.9666966696669667, 0.9666666666666667, 0.9666666666666667, 0.9684684684684685, 0.9576576576576576],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['TF-IDF-KNN']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Prep/Q']*num_results,
                                  'F1': [0.8775510204081634, 0.8467741935483872, 0.8918032786885246, 0.8412698412698412, 0.7946127946127947],
                                  'Accuracy': [0.9675967596759676, 0.9657657657657658, 0.9702702702702702, 0.963963963963964, 0.945045045045045],
                                  })
df_results = df_results.append(df_single_results)

# EMAILS
# Methods
df_single_results = pd.DataFrame({'Model': ['TF-IDF-KNN']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Raw/BOW']*num_results,
                                  'F1': [0.8969521044992743, 0.9079939668174962, 0.9009584664536742, 0.9002932551319649, 0.8784722222222222],
                                  'Accuracy': [0.9355716878402904, 0.9446460980036298, 0.9436875567665758, 0.9382379654859219, 0.9364214350590372],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['TF-IDF-KNN']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Raw/Q']*num_results,
                                  'F1': [0.8575418994413408, 0.9217134416543574, 0.8348348348348348, 0.8618784530386739, 0.8216039279869067],
                                  'Accuracy': [0.9074410163339383, 0.9519056261343013, 0.9000908265213442, 0.9091734786557675, 0.9009990917347865],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['TF-IDF-KNN']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Prep/BOW']*num_results,
                                  'F1': [0.9384164222873901, 0.9444444444444445, 0.9238249594813615, 0.9276218611521418, 0.9309090909090909],
                                  'Accuracy': [0.9618874773139746, 0.9673321234119783, 0.9573115349682108, 0.9554950045413261, 0.9654859218891917],
                                  })
df_results = df_results.append(df_single_results)

df_single_results = pd.DataFrame({'Model': ['TF-IDF-KNN']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Prep/Q']*num_results,
                                  'F1': [0.9356725146198831, 0.9331259720062209, 0.9173419773095624, 0.9571865443425076, 0.9477611940298508],
                                  'Accuracy': [0.9600725952813067, 0.9609800362976406, 0.9536784741144414, 0.9745685740236149, 0.9745685740236149],
                                  })
df_results = df_results.append(df_single_results)



# -- TF-IDF-NB --

# SMS
# Methods
df_single_results = pd.DataFrame({'Model': ['TF-IDF-NB']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Raw/BOW']*num_results,
                                  'F1': [0.9054054054054055, 0.9019607843137254, 0.8996763754045307, 0.8593750000000001, 0.8451612903225807],
                                  'Accuracy': [0.9747974797479748, 0.9774774774774775, 0.972072072072072, 0.9675675675675676, 0.9567567567567568],
                                  })
df_results = df_results.append(df_single_results)
df_single_results = pd.DataFrame({'Model': ['TF-IDF-NB']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Raw/Q']*num_results,
                                  'F1': [0.9246575342465755, 0.921259842519685, 0.9389067524115756, 0.8809523809523809, 0.87248322147651],
                                  'Accuracy': [0.9801980198019802, 0.9819819819819819, 0.9828828828828828, 0.972972972972973, 0.9657657657657658],
                                  })
df_results = df_results.append(df_single_results)
df_single_results = pd.DataFrame({'Model': ['TF-IDF-NB']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Prep/BOW']*num_results,
                                  'F1': [0.855457227138643, 0.8322147651006712, 0.8653295128939827, 0.8106312292358804, 0.84],
                                  'Accuracy': [0.9558955895589559, 0.954954954954955, 0.9576576576576576, 0.9486486486486486, 0.9495495495495495],
                                  })
df_results = df_results.append(df_single_results)
df_single_results = pd.DataFrame({'Model': ['TF-IDF-NB']*num_results,
                                  'Dataset': ['SMS']*num_results,
                                  'Method': ['Prep/Q']*num_results,
                                  'F1': [0.8666666666666667, 0.833922261484099, 0.8716417910447762, 0.7896440129449839, 0.8459214501510574],
                                  'Accuracy': [0.9603960396039604, 0.9576576576576576, 0.9612612612612612, 0.9414414414414415, 0.9540540540540541],
                                  })
df_results = df_results.append(df_single_results)

# EMAILS
# Methods
df_single_results = pd.DataFrame({'Model': ['TF-IDF-NB']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Raw/BOW']*num_results,
                                  'F1': [0.9341692789968652, 0.9478827361563518, 0.9401408450704225, 0.9592476489028212, 0.937381404174573],
                                  'Accuracy': [0.9618874773139746, 0.9709618874773139, 0.9691189827429609, 0.9763851044504995, 0.9700272479564033],
                                  })
df_results = df_results.append(df_single_results)
df_single_results = pd.DataFrame({'Model': ['TF-IDF-NB']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Raw/Q']*num_results,
                                  'F1': [0.8932676518883415, 0.9081632653061225, 0.9064220183486238, 0.9018302828618968, 0.8658536585365854],
                                  'Accuracy': [0.941016333938294, 0.9509981851179673, 0.9536784741144414, 0.9464123524069028, 0.9400544959128065],
                                  })
df_results = df_results.append(df_single_results)
df_single_results = pd.DataFrame({'Model': ['TF-IDF-NB']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Prep/BOW']*num_results,
                                  'F1': [0.9616519174041298, 0.9708141321044546, 0.9356913183279743, 0.9506726457399104, 0.948306595365419],
                                  'Accuracy': [0.9764065335753176, 0.9827586206896551, 0.963669391462307, 0.9700272479564033, 0.9736603088101726],
                                  })
df_results = df_results.append(df_single_results)
df_single_results = pd.DataFrame({'Model': ['TF-IDF-NB']*num_results,
                                  'Dataset': ['EMAILS']*num_results,
                                  'Method': ['Prep/Q']*num_results,
                                  'F1': [0.9483568075117371, 0.9647435897435898, 0.9661016949152542, 0.9591194968553458, 0.9550561797752809],
                                  'Accuracy': [0.97005444646098, 0.9800362976406534, 0.9818346957311535, 0.9763851044504995, 0.9782016348773842],
                                  })
df_results = df_results.append(df_single_results)


df_results.to_csv('results.csv', index=False)







