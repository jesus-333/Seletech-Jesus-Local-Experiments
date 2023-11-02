This readme file was generated on 25-07-2023 by Dagmawi D. Tegegn

GENERAL INFORMATION

Title: Beans Second Phase Experiment

Date of Data Collection: 27-06-2023 to 18-07-2023

Geographic Location of Data Collection: Università degli Studi di Milano - Bicocca, 2nd floor lab (u3-2nd floor lab)

DATA & FILE OVERVIEW

File List:
This folder contains data from various days of acquisition grouped from t0 to t6, representing the time points described in the bean-setup-protocol_EP_v2.pdf file.
Each time point contains five main folders: [csv, insight2, media, miflora, temp_humidity]. An example is shown below.

+---t0
¦   +---csv
¦   +---insight2
¦   ¦   +---PhaseolusVulgaris
¦   ¦   +---ViciaFaba
¦   +---media
¦   +---miflora
¦   +---temp_humidity
+---t1
+---t2
+---t3
+---t4
+---t5
+---t6

- csv folder: Contains the csv files of the spectra parsed: beans.csv and beans_avg.csv, representing all the data and the averaged data on the three consecutive spectra per parameter, respectively.
- insight2 folder: Contains the .ard files from the spectrometer of the two plants. The file name structure for each plant is as follows:
  plantLabel_YYYY_MM_DD_h_m_s i.e CON1_2023_06_27_12_25_19
- media folder: Contains photographic evidence of the setup for both plants.
- miflora folder: Contains the data from the two MiFlora devices labeled "GreenMiFlora" and "WhiteMiFlora".
- temp_humidity folder: Contains the data from the temperature and humidity sensor.

The beans.csv & beans_avg.csv have the following attributes:
- timestamp: YYYY-MM-DD h:m:s
- [1350-2150]: wavelength range
- device_id: the spectrometer used
- path: path of the .ard file parsed
- plant: plants analyzed ViciaFaba and PhaseolusVulgaris
- type: the label of each plant leaf [CON1, CON2,..., CON5], [NACL150_1, NACL150_2,..., NACL150_5], [NACL300_1, NACL300_2,..., NACL300_5]
- test_control: label to group the data into control, test_150, and test_300
- gain_0, gain_1: values of gain for each mems of the spectrometer
- lamp_0, lamp_1: lamp power values for the two lamps of the spectrometer.
- NTC AVG [°C]: average temperature of the two mems of the spectrometer

MiFlora csv:
- TIMESTAMP: YYYY-MM-DD h:m:s (*not always the seconds are available)
- MI_BATTERY: battery status of the device (%)
- DEVICEID: label of the two devices (GreenMiFlora and WhiteMiFlora)
- DEVICEADDRS: MAC address of the devices
- FWV: Firmware version of the devices
- MI_CONDUCTIVITY: conductivity value (µS/cm)
- MI_LIGHT: light value (lux)
- MI_MOISTURE: moisture values (%)
- MI_TEMPERATURE: temperature value (°C)

Note: The MiFlora devices are launched from a Python script [Seletech]MiFlora_1.3.py, and data is read continuously. Therefore, the time range where the values are not considered is when transitioning from one plant to another. To ensure accurate time for each leaf, the timestamp from beans.csv and the timestamp from MiFlora must match. In timepoint t5, MiFlora was interrupted (check note_t5.txt). The values must be interpreted with caution.

temp_humidity csv:
- Timestamp: YYYY_MM_DD_h_m_s (*has not yet been parsed)
- Temperature: temperature value [°C]
- Humidity: humidity value [%]

Note: The temp_humidity sensor is launched from its custom-made application, and data is read continuously every 1000ms. It is fixed at a precise spot in the setup for the whole duration of the acquisition process.

General Note:
The acquisition process always started from the control group of the Vicia Faba plant, followed by the test group of the same plant, and lastly, the Phaseolus Vulgaris. The plant leaves were not always in the same morphological state between time points. Refer to the photos. We cannot negate that there could be interference from the environment.
