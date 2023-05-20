# Rebalancing Methods Performance Results

The following table lists dataset pairs for IRDP and CPDP scenarios:

Training → Testing Datasets for IRDP Scenario | Training → Testing Datasets for CRDP Scenario |
--- | --- 
Ant 1.3 → 1.4, 1.5, 1.6, 1.7 <br/>  Ant 1.4 → 1.5, 1.6, 1.7 <br/>  Ant 1.5 → 1.6, 1.7 <br/>  Ant 1.6 → 1.7 | JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3 + Lucene 2.0 + pBeans 2.0 + Poi 2.0RC1 + Synapse 1.1, 1.2 + Velocity 1.6.1 + Xalan 2.6 + Xerces 1.2, 1.3 → Ant 1.3, 1.4, 1.5, 1.6, 1.7
JEdit 3.2.1 → 4.0, 4.1, 4.2, 4.3 <br/> JEdit 4.0 → 4.1, 4.2, 4.3 <br/> JEdit 4.1 → 4.2, 4.3 <br/> JEdit 4.2 → 4.3 | Ant 1.3, 1.4, 1.5, 1.6, 1.7 + Lucene 2.0 + pBeans 2.0 + Poi 2.0RC1 + Synapse 1.1, 1.2 + Velocity 1.6.1 + Xalan 2.6 + Xerces 1.2, 1.3 → JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3
Lucene 2.0 → 2.2, 2.4 | Ant 1.3, 1.4, 1.5, 1.6, 1.7 + JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3 + pBeans 2.0 + Poi 2.0RC1 + Synapse 1.1, 1.2 + Velocity 1.6.1 + Xalan 2.6 + Xerces 1.2, 1.3 → Lucene 2.0, 2.2, 2.4
 <br/> | Ant 1.3, 1.4, 1.5, 1.6, 1.7 + JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3 + Lucene 2.0 + Poi 2.0RC1 + Synapse 1.1, 1.2 + Velocity 1.6.1 + Xalan 2.6 + Xerces 1.2, 1.3 → pBeans 1.0, 2.0
Poi 2.0RC1 → 2.5.1, 3.0 | Ant 1.3, 1.4, 1.5, 1.6, 1.7 + JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3 + Lucene 2.0 + pBeans 2.0 + Synapse 1.1, 1.2 + Velocity 1.6.1 + Xalan 2.6 + Xerces 1.2, 1.3 → Poi 1.5, 2.0RC1, 2.5.1, 3.0
Synapse 1.1 → 1.2 | Ant 1.3, 1.4, 1.5, 1.6, 1.7 + JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3 + Lucene 2.0 + pBeans 2.0 + Poi 2.0RC1 + Velocity 1.6.1 + Xalan 2.6 + Xerces 1.2, 1.3 → Synapse 1.0, 1.1, 1.2 
 <br/> | Ant 1.3, 1.4, 1.5, 1.6, 1.7 + JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3 + Lucene 2.0 + pBeans 2.0 + Poi 2.0RC1 + Synapse 1.1, 1.2 + Xalan 2.6 + Xerces 1.2, 1.3 → Velocity 1.4, 1.5, 1.6.1
Xalan 2.6 → 2.7 |  Ant 1.3, 1.4, 1.5, 1.6, 1.7 + JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3 + Lucene 2.0 + pBeans 2.0 + Poi 2.0RC1 + Synapse 1.1, 1.2 + Velocity 1.6.1 + Xerces 1.2, 1.3 → Xalan 2.6, 2.7 
Xerces 1.2 → 1.3 | Ant 1.3, 1.4, 1.5, 1.6, 1.7 + JEdit 3.2.1, 4.0, 4.1, 4.2, 4.3 + Lucene 2.0 + pBeans 2.0 + Poi 2.0RC1 + Synapse 1.1, 1.2 + Xalan 2.6 + Velocity 1.6.1 → Xerces 1.2, 1.3


* For IRDP scenario:

    Training dataset =  ("Project" column) + ("Train Version" column)

    Testing dataset = ("Project" column) + ("Test Version" column)


* For CPDP sceneario:

    Training dataset = (All versions of all projects) - ("Project" column's versions)

    Testing dataset = ("Project" column) + ("Test Version" column)