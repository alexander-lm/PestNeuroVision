# PestNeuroVision: A mobile application based on convolutional neural networks (CNNs) and computer vision for the detection of agricultural pests in the Cañete Valley, Lima, Peru. 🐛📱

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Model: YOLO11s](https://img.shields.io/badge/Model-YOLO11s-red?logo=ultralytics&logoColor=white)](https://ultralytics.com/)
[![IDE: Android Studio](https://img.shields.io/badge/IDE-Android%20Studio-3DDC84?logo=android-studio&logoColor=white)](https://developer.android.com/studio)
[![Language: Kotlin](https://img.shields.io/badge/Language-Kotlin-7F52FF?logo=kotlin&logoColor=white)](https://kotlinlang.org/)

**PestNeuroVision** is an Android mobile application for detecting agricultural pests using convolutional neural networks and computer vision. Developed as part of a scientific research and technological innovation project, it integrates the YOLO11s object detection model—which runs entirely on-device without requiring an internet connection—to identify various species.

<hr style="height:4px; border:none; background-color:#444;">

## 📱 Features

- Detection via camera or photo gallery.
- Confidence level for each detection.
- Detection history.
- Statistical charts of detections.
- Technical information for each species.
- 100% offline operation — no internet connection required.

<hr style="height:4px; border:none; background-color:#444;">

## 🏷️ Detectable pests

| Species | Life Stage |
|---|---|
| *Bemisia tabaci* | Adult |
| *Ceratitis capitata* | Adult |
| *Dione juno* | Adult |
| *Dione juno* | Larva |
| *Ligyrus gibbosus* | Adult |
| *Liriomyza huidobrensis* | Adult |
| *Myzus persicae* | Nymph |
| *Spodoptera frugiperda* | Adult |
| *Spodoptera frugiperda* | Larva |

<hr style="height:4px; border:none; background-color:#444;">

### 📊 Model performance (YOLO11s)

| Metrics | Value |
|---|---|
| Precision | 92.4 % |
| Recall | 87.7 % |
| mAP@50 | 91.7 %|
| mAP@50-95 | 78.0 % |

<hr style="height:4px; border:none; background-color:#444;">

## 📱 Mobile App Development (Android Studio)

### 🛠️ Technology Stack

* **Language:** Kotlin
* **Architecture:** Model-View-ViewModel (MVVM)
* **IA/DL:** TensorFlow Lite
* **Database:** SQLite and Room Persistence Library 
* **Graphs:** MPAndroidChart

---

### 📋 Development environment requirements (developer)

- **Operating System:** Windows 10 or higher, Linux Ubuntu 22.04 LTS or higher, or macOS 13 or higher.
- **Processor:** Intel Core i5 or equivalent.
- **RAM:** 8 GB or higher.
- **Storage capacity:** 10 GB or higher.
- **IDE:** Android Studio Narwhal v2025.1.1 or higher.
- **Physical device:** Recommended (the application uses the device's camera and gallery).
- **Emulator (Optional):** Android Virtual Device (AVD), included in Android Studio. Functional for gallery testing, with limited camera support.

---

### 📋 Minimum mobile device requirements (end user)

- **Operating System:** Android 11 (API 30) or higher.
- **Processor:** Octa-core 2.0 GHz or higher.
- **RAM:** 4 GB or higher.
- **Storage capacity:** 150 MB minimum.

---

### 📂 Project Structure
Simplified structure of the MVVM pattern used by the application:

```text
├── data/           # Data Layer: Local persistence (Room/DAOs) and external data sources.
├── repository/     # Data Management: Data source abstraction and implementation of TFLite inference.
├── ui/             # Presentation Layer: Interface components (Activities/Fragments) and graphics rendering.
└── viewmodel/      # Presentation Logic: Managing the UI state and communicating with the repository.
```

---

## ⚙️ Import, configure, and run the source code

1. **Clone the repository:**
```bash
   git clone https://github.com/alexander-lm/PestNeuroVision.git
```

2. **Open the project in Android Studio:** Select the cloned repository.

3. **Sync Gradle:** Android Studio will automatically sync the dependencies when the project is opened. Otherwise, select _File > Sync Project with Gradle Files_. The project uses Kotlin v2.0.21 with updated dependencies.

4. **Configure the device:**
   - **Physical device (recommended):**
     - Enable **Developer Options** from the device settings.
     - Access **Developer Options** and select the debugging method:
       - **USB cable:** Enable **USB Debugging**.
       - **Wi-Fi:** Enable **Wireless Debugging** and make sure both the computer and the device are connected to the same network.
   - **Emulator (optional):**
     - Set up a virtual device from _Device Manager > Create Virtual Device_ in Android Studio.

5. **Run the application (Shift + F10):** Select the target device and run the application.

<hr style="height:4px; border:none; background-color:#444;">

### 🔐 Application Login Credentials

Use the following credentials to log in to the PestNeuroVision app:

- **User:** admin
- **Password:** admin

---

### 📊 Application Modules

* **Pest Detection:** An interface for detecting agricultural pests through image analysis, using images either selected from the gallery or captured with the device's camera. Compatible with JPEG, JPG, PNG, and WebP formats.
* **Detection History:** An interface displaying a list of detected pests. It includes a search filter based on the scientific name of the species.
* **Insect:** An interface displaying the pest catalog. It includes a search filter based on the pest's scientific name. You can view technical information about each species by tapping on the insect's image.
* **Statistical Graphs:** An interface for the visual analysis of fluctuations in the volume of detected pests. It includes three components: line, bar, and pie charts.
* **My Account:** Interface for configuring user data.
* **About:** An interface that displays the application's version and license information, the model, and the credits for the images included in the application.

---

## 📌 Recommendations for use

For best detection results, we recommend using square images (height = width), as the model (YOLO11s) was trained on images with a resolution of 640 × 640 pixels.

<hr style="height:4px; border:none; background-color:#444;">

## 👤 Development Team

**Lead Developer and Researcher:** Alexander Leandro Mendoza — 🔗 **ORCID:** [0000-0002-8514-6804](https://orcid.org/0000-0002-8514-6804)\
**Academic Advisor:** Alex Abelardo Pacheco Pumaleque - **ORCID:** [0000-0001-9721-0730](https://orcid.org/0000-0001-9721-0730)\
**Funding Entity:** Directorate of Innovation and Technology Transfer (DITT) — Universidad Nacional de Cañete (UNDC)

<hr style="height:4px; border:none; background-color:#444;">

## 🎖️ Credits

The image credits for the images used in this project are listed in the `image_credits.csv` file.

<hr style="height:4px; border:none; background-color:#444;">

## ⚖️ License

This project is distributed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0), with the exception of the images, which retain their original licenses (CC0 1.0 / CC BY 4.0). The fine-tuned YOLO11s weights used in this project, as well as the resulting metrics, are derivative works of the 
Ultralytics YOLO11 base model, which is also governed by the [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) license. Any modifications or derivative works must also be distributed under the same license, with the source code made publicly available.


[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

<hr style="height:4px; border:none; background-color:#444;">

## 💡 Technical Disclaimer

⚠️ PestNeuroVision is a tool designed to assist in the detection of agricultural pests. The app is not intended to replace the judgment of an expert. It is always recommended to have the results verified by an agricultural professional.

<hr style="height:4px; border:none; background-color:#444;">

## 📧 Contact
**Email:** leandro.alexander.2022@gmail.com\
**University:** [Universidad Nacional de Cañete (UNDC)](https://web.undc.edu.pe/)\
**GitHub:** [https://github.com/alexander-lm/PestNeuroVision](https://github.com/alexander-lm/PestNeuroVision)
