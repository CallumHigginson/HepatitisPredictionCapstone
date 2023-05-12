import tkinter as tk
from Hepatitis_Predictor_Model import *


class CVD_GUI:
    def __init__(self):

        # Create the main window.
        self.main_window = tk.Tk()
        self.main_window.title("Hepatitis Survival Prediction Tool")

        # Create two frames to group widgets.
        self.one_frame = tk.Frame()
        self.two_frame = tk.Frame()
        self.three_frame = tk.Frame()
        self.four_frame = tk.Frame()
        self.five_frame = tk.Frame()
        self.six_frame = tk.Frame()
        self.seven_frame = tk.Frame()
        self.eight_frame = tk.Frame()
        self.nine_frame = tk.Frame()
        self.ten_frame = tk.Frame()
        self.eleven_frame = tk.Frame()
        self.twelve_frame = tk.Frame()
        self.thirteen_frame = tk.Frame()
        self.fourteen_frame = tk.Frame()
        self.fifteen_frame = tk.Frame()
        self.sixteen_frame = tk.Frame()
        self.seventeen_frame = tk.Frame()
        self.eighteen_frame = tk.Frame()
        self.nineteen_frame = tk.Frame()
        self.twenty_frame = tk.Frame()
        self.twenty_one_frame = tk.Frame()
        self.twenty_two_frame= tk.Frame()

        # Create the widgets for one frame. (title display)
        self.title_label = tk.Label(self.one_frame, text='HEPATITIS SURVIVAL PREDICTION TOOL',fg="Blue",
                                    font=("Helvetica", 18))
        self.title_label.pack()

        # Create the widgets for two frame. (age input)
        self.age_label = tk.Label(self.two_frame, text='Age:')
        self.age_entry = tk.Entry(self.two_frame, bg="white", fg="black", width = 10)
        self.age_entry.insert(0,'0')
        self.age_label.pack(side='left')
        self.age_entry.pack(side='left')

        # Create the widgets for three frame. (sex/gender input)
        self.sex_label = tk.Label(self.three_frame, text='Sex:')
        self.click_sex_var = tk.StringVar()
        self.click_sex_var.set("Male")
        self.sex_inp = tk.OptionMenu(self.three_frame,self.click_sex_var, "Male", "Female")
        self.sex_label.pack(side='left')
        self.sex_inp.pack(side='left')

        # Create the widgets for four frame. (steroid input)
        self.steroid_label = tk.Label(self.four_frame, text='Steroid:')
        self.click_steroid_var = tk.StringVar()
        self.click_steroid_var.set("No")
        self.steroid_inp = tk.OptionMenu(self.four_frame, self.click_steroid_var, "No", "Yes")
        self.steroid_label.pack(side='left')
        self.steroid_inp.pack(side='left')

        # Create the widgets for five frame. (antivirals input)
        self.antivirals_label = tk.Label(self.five_frame, text='Antivirals:')
        self.click_antivirals_var = tk.StringVar()
        self.click_antivirals_var.set("No")
        self.antivirals_inp = tk.OptionMenu(self.five_frame, self.click_antivirals_var, "No", "Yes")
        self.antivirals_label.pack(side='left')
        self.antivirals_inp.pack(side='left')

        # Create the widgets for six frame. (fatigue  input)
        self.fatigue_label = tk.Label(self.six_frame, text='Fatigue:')
        self.click_fatigue_var = tk.StringVar()
        self.click_fatigue_var.set("No")
        self.fatigue_inp = tk.OptionMenu(self.six_frame, self.click_fatigue_var, "No", "Yes")
        self.fatigue_label.pack(side='left')
        self.fatigue_inp.pack(side='left')

        # Create the widgets for seven frame. (malaise input)
        self.malaise_label = tk.Label(self.seven_frame, text='Malaise:')
        self.click_malaise_var = tk.StringVar()
        self.click_malaise_var.set("No")
        self.malaise_inp = tk.OptionMenu(self.seven_frame, self.click_malaise_var, "No", "Yes")
        self.malaise_label.pack(side='left')
        self.malaise_inp.pack(side='left')

        # Create the widgets for eight frame. (anorexia  input)
        self.anorexia_label = tk.Label(self.eight_frame, text='Anorexia:')
        self.click_anorexia_var = tk.StringVar()
        self.click_anorexia_var.set("No")
        self.anorexia_inp = tk.OptionMenu(self.eight_frame, self.click_anorexia_var, "No", "Yes")
        self.anorexia_label.pack(side='left')
        self.anorexia_inp.pack(side='left')

        # Create the widgets for nine frame. (liver big input)
        self.liver_big_label = tk.Label(self.nine_frame, text='Liver Big:')
        self.click_liver_big_var = tk.StringVar()
        self.click_liver_big_var.set("No")
        self.liver_big_inp = tk.OptionMenu(self.nine_frame, self.click_liver_big_var, "No", "Yes")
        self.liver_big_label.pack(side='left')
        self.liver_big_inp.pack(side='left')

        # Create the widgets for ten frame. (liver firm  input)
        self.liver_firm_label = tk.Label(self.ten_frame, text='Liver Firm:')
        self.click_liver_firm_var = tk.StringVar()
        self.click_liver_firm_var.set("No")
        self.liver_firm_inp = tk.OptionMenu(self.ten_frame, self.click_liver_firm_var, "No", "Yes")
        self.liver_firm_label.pack(side='left')
        self.liver_firm_inp.pack(side='left')

        # Create the widgets for eleven frame (spleen palpable input)
        self.spleen_palpable_label = tk.Label(self.eleven_frame, text='Spleen Palpable:')
        self.click_spleen_palpable_var = tk.StringVar()
        self.click_spleen_palpable_var.set("No")
        self.spleen_palpable_inp = tk.OptionMenu(self.eleven_frame, self.click_spleen_palpable_var, "No", "Yes")
        self.spleen_palpable_label.pack(side='left')
        self.spleen_palpable_inp.pack(side='left')

        # Create the widgets for twelve frame (spider angiomas input)
        self.spiders_label = tk.Label(self.twelve_frame, text='Spider angiomas:')
        self.click_spiders_var = tk.StringVar()
        self.click_spiders_var.set("No")
        self.spiders_inp = tk.OptionMenu(self.twelve_frame, self.click_spiders_var, "No", "Yes")
        self.spiders_label.pack(side='left')
        self.spiders_inp.pack(side='left')

        # Create the widgets for thirteen frame (ascites input)
        self.ascites_label = tk.Label(self.thirteen_frame, text='Ascites:')
        self.click_ascites_var = tk.StringVar()
        self.click_ascites_var.set("No")
        self.ascites_inp = tk.OptionMenu(self.thirteen_frame, self.click_ascites_var, "No", "Yes")
        self.ascites_label.pack(side='left')
        self.ascites_inp.pack(side='left')

        # Create the widgets for fourteen frame (esophageal varices input)
        self.varices_label = tk.Label(self.fourteen_frame, text='Esophageal varices:')
        self.click_varices_var = tk.StringVar()
        self.click_varices_var.set("No")
        self.varices_inp = tk.OptionMenu(self.fourteen_frame, self.click_varices_var, "No", "Yes")
        self.varices_label.pack(side='left')
        self.varices_inp.pack(side='left')

        # Create the widgets for fifteen frame (ALK phosphatase input)
        self.alk_phosphate_label = tk.Label(self.fifteen_frame, text='ALK Phosphatase amount:')
        self.alk_phosphate_entry = tk.Entry(self.fifteen_frame, bg="white", fg="black")
        self.alk_phosphate_entry.insert(0,'0')
        self.alk_phosphate_label.pack(side='left')
        self.alk_phosphate_entry.pack(side='left')

        # Create the widgets for sixteen frame (Bilirubin input)
        self.bilirubin_label = tk.Label(self.sixteen_frame, text='Bilirubin amount:')
        self.bilirubin_entry = tk.Entry(self.sixteen_frame, bg="white", fg="black")
        self.bilirubin_entry.insert(0,'0')
        self.bilirubin_label.pack(side='left')
        self.bilirubin_entry.pack(side='left')

        # Create the widgets for seventeen frame (SGOT input)
        self.sgot_label = tk.Label(self.seventeen_frame, text='AST (SGOT) amount:')
        self.sgot_entry = tk.Entry(self.seventeen_frame, bg="white", fg="black")
        self.sgot_entry.insert(0,'0')
        self.sgot_label.pack(side='left')
        self.sgot_entry.pack(side='left')

        # Create the widgets for eighteen frame (Albumin input)
        self.albumin_label = tk.Label(self.eighteen_frame, text='Albumin amount:')
        self.albumin_entry = tk.Entry(self.eighteen_frame, bg="white", fg="black")
        self.albumin_entry.insert(0,'0')
        self.albumin_label.pack(side='left')
        self.albumin_entry.pack(side='left')

        # Create the widgets for nineteen frame (Protime input)
        self.protime_label = tk.Label(self.nineteen_frame, text='Prothrombim test time:')
        self.protime_entry = tk.Entry(self.nineteen_frame, bg="white", fg="black")
        self.protime_entry.insert(0,'0')
        self.protime_label.pack(side='left')
        self.protime_entry.pack(side='left')

        # Create the widgets for twenty frame (Histology input)
        self.histology_label = tk.Label(self.twenty_frame, text='Histology done:')
        self.click_histology_var = tk.StringVar()
        self.click_histology_var.set("No")
        self.histology_inp = tk.OptionMenu(self.twenty_frame, self.click_histology_var, "No", "Yes")
        self.histology_label.pack(side='left')
        self.histology_inp.pack(side='left')


        # Create the widgets for twenty two frame (display prediction)
        self.hepatitis_predict_ta = tk.Text(self.twenty_two_frame, height=20, width=45, bg='light blue')

        #Create predict button and quit button
        self.btn_predict = tk.Button(self.twenty_one_frame, text='Predict Heart Disease', command=self.predict_hepatitis)
        self.btn_quit = tk.Button(self.twenty_one_frame, text='Quit', command=self.main_window.destroy)

        self.hepatitis_predict_ta.pack(side='left')
        self.btn_predict.pack()
        self.btn_quit.pack()

        # Pack the frames.
        self.one_frame.pack()
        self.two_frame.pack()
        self.three_frame.pack()
        self.four_frame.pack()
        self.five_frame.pack()
        self.six_frame.pack()
        self.seven_frame.pack()
        self.eight_frame.pack()
        self.nine_frame.pack()
        self.ten_frame.pack()
        self.eleven_frame.pack()
        self.twelve_frame.pack()
        self.thirteen_frame.pack()
        self.fourteen_frame.pack()
        self.fifteen_frame.pack()
        self.sixteen_frame.pack()
        self.seventeen_frame.pack()
        self.eighteen_frame.pack()
        self.nineteen_frame.pack()
        self.twenty_frame.pack()
        self.twenty_one_frame.pack()
        self.twenty_two_frame.pack()

        # Enter the tkinter main loop.
        tk.mainloop()

    def predict_hepatitis(self):
        result_string = ""

        self.hepatitis_predict_ta.delete(0.0, tk.END)
        patient_age = self.age_entry.get()
        patient_age = int(patient_age)

        patient_sex = self.click_sex_var.get()
        if(patient_sex == "Male"):
            patient_sex = 0
        else:
            patient_sex = 1

        patient_steroid = self.click_steroid_var.get()
        if(patient_steroid == "No"):
            patient_steroid = 0
        else:
            patient_steroid = 1

        patient_antivirals = self.click_antivirals_var.get()
        if(patient_antivirals == "No"):
            patient_antivirals = 0
        else:
            patient_antivirals = 1

        patient_fatigue = self.click_fatigue_var.get()
        if(patient_fatigue == "No"):
            patient_fatigue = 0
        else:
            patient_fatigue = 1

        patient_malaise = self.click_malaise_var.get()
        if(patient_malaise == "No"):
            patient_malaise = 0
        else:
            patient_malaise = 1

        patient_anorexia = self.click_anorexia_var.get()
        if(patient_anorexia == "No"):
            patient_anorexia = 0
        else:
            patient_anorexia = 1

        patient_liver_big = self.click_liver_big_var.get()
        if(patient_liver_big == "No"):
            patient_liver_big = 0
        else:
            patient_liver_big = 1

        patient_liver_firm = self.click_liver_firm_var.get()
        if(patient_liver_firm == "No"):
            patient_liver_firm = 0
        else:
            patient_liver_firm = 1

        patient_spleen_palpable = self.click_spleen_palpable_var.get()
        if(patient_spleen_palpable == "No"):
            patient_spleen_palpable = 0
        else:
            patient_spleen_palpable = 1

        patient_spiders = self.click_spiders_var.get()
        if(patient_spiders == "No"):
            patient_spiders = 0
        else:
            patient_spiders = 1

        patient_ascites = self.click_ascites_var.get()
        if(patient_ascites == "No"):
            patient_ascites = 0
        else:
            patient_ascites = 1

        patient_varices = self.click_varices_var.get()
        if(patient_varices == "No"):
            patient_varices = 0
        else:
            patient_varices = 1

        patient_histology = self.click_histology_var.get()
        if(patient_histology == "No"):
            patient_histology = 0
        else:
            patient_histology = 1

        patient_bilirubin = self.bilirubin_entry.get()
        patient_alk_phosphate = self.alk_phosphate_entry.get()
        patient_sgot = self.sgot_entry.get()
        patient_albumin = self.albumin_entry.get()
        patient_protime = self.protime_entry.get()

        patient_bilirubin = float(patient_bilirubin)
        patient_alk_phosphate = float(patient_alk_phosphate)
        patient_sgot = float(patient_sgot)
        patient_albumin = float(patient_albumin)
        patient_protime = float(patient_protime)

        result_string += "===Patient Diagnosis=== \n"
        patient_info = (patient_age,patient_sex,patient_steroid, patient_antivirals,patient_fatigue,
                         patient_malaise,patient_anorexia, patient_liver_big, patient_liver_firm,
                         patient_spleen_palpable, patient_spiders, patient_ascites, patient_varices, patient_bilirubin,
                         patient_alk_phosphate, patient_sgot,patient_albumin, patient_protime, patient_histology)


        hepatitis_prediction =  best_model.predict([patient_info])
        disp_string = (f"This prediction has an accuracy of: {model_accuracy *100:.2f}%\n")

        print(hepatitis_prediction)
        result = hepatitis_prediction


        if(hepatitis_prediction == [1]):
            result_string += disp_string + '\n' + "1 - You are likely to survive hepatitis."
        else:
            result_string += disp_string + '\n' + "2 - Your current health condition places \nyou at an elevated risk for potentially fatal complications associated with hepatitis. \n Please consult your GP."
        self.hepatitis_predict_ta.insert('1.0',result_string)

        # Predicted:  1 Actual:  1 Data:  (16, 0, 1, 14, 74, 0, 1, 61, 0, 11, 2, 0, 2)
        # Predicted:  0 Actual:  0 Data:  (35, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 29, 30, 29, 9, 4, 1)

my_cvd_GUI = CVD_GUI()
