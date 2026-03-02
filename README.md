# MDA-CSI and DFS-CSI

This repository contains the proposed **MDA-CSI** methodology, designed to identify human activity in a room through Channel State Information (CSI) data analysis. The model was evaluated using a dataset of 59 volunteers.

This repository also includes an extended version called **DFS-CSI**, which was evaluated on a larger dataset of 86 volunteers.


---

### 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

### 📊 Data Availability & Processing

* **Raw & Statistical Data:** The **raw CSI data** and the **statistical social data** from the dataset are available for research purposes and can be shared **upon request**.
* **Large Files:** Due to GitHub's file size limitations, some large files (specifically the training dataloaders and pre-processed data) are hosted externally:
**MDA-CSI**
* **[Download Dataset & Dataloaders (Google Drive)](https://drive.google.com/drive/folders/1ze26yuKVfrVItPifkPjPCfkEfP4HJzE2?usp=sharing)**

---

### 🚀 How to Run

To run the transformer model, follow these steps:

1. **Download** this repository and the external dataset files.
2. **Install** the environment using Anaconda:
```bash
conda env create -f CSI_Windows_selenium_v2.yml
conda activate csi

```


3. **Run** the test file:
```bash
python test4amplitude48canais.py

```

**Data Unpacking and Training**
If you want to unpack the raw CSI data, train, and evaluate the model from scratch, you can use the provided example script:

```bash
python Lan_prepro.py

```

---

### 📝 Publications & Citation

If you use this code or dataset in your research, please cite the following papers:

**MDA-CSI Methodology (WD 2025):**

```bibtex
@INPROCEEDINGS{11302651,
  author={Dos Santos, Allan Costa Nascimento and others},
  booktitle={2025 13th Wireless Days Conference (WD)}, 
  title={A Transformer-Based Methodology for Person-Independent Human Activity Recognition Using Wi-Fi Csi}, 
  year={2025},
  pages={1-9},
  doi={10.1109/WD67713.2025.11302651}}

```

**Complementary Work (HealthCom 2024):**

```bibtex
@INPROCEEDINGS{10880838,
  author={dos Santos, Allan Costa Nascimento and others},
  booktitle={2024 IEEE International Conference on E-health Networking, Application \& Services (HealthCom)}, 
  title={A Computer Vision Model to Support Individuals with Disabilities Within University Campuses}, 
  year={2024},
  pages={1-7},
  doi={10.1109/HealthCom60970.2024.10880838}}

```

---

### 📧 Contact

For access to raw data, statistical social data, or further inquiries:

**Allan Costa Nascimento dos Santos**

*Ph.D. Student / Visiting Doctoral Researcher*

Universidade Federal Fluminense (UFF) / Brunel University London

📩 [allans@midiacom.uff.br](mailto:allans@midiacom.uff.br) | [Allan.Santos@brunel.ac.uk](mailto:Allan.Santos@brunel.ac.uk)


====================

Jesus answered, ‘I am the way and the truth and the life. No-one comes to the Father except through me.

John 14:6 NIVUK

https://bible.com/bible/129/jhn.14.6.NVI

allancostans.blogspot.com 


