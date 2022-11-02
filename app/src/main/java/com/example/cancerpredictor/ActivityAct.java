package com.example.cancerpredictor;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import static weka.core.SerializationHelper.read;

public class ActivityAct extends AppCompatActivity  implements SensorEventListener {

    private Button train_b,deploy_b;
    private Button start_collect,end_collect;

    private boolean record=false;
    private EditText edit_label;

    Classifier nb,tree,lmt;

    private TextView[] deploy_text=new TextView[3];
    private TextView[] train_text=new TextView[3];

    private static final String filename="Nazim_activity.csv";
    //use this variable to differentiate between tasks
    // code =0 ==> Collect
    // code 2 ==> Deploy
    static byte code=1,j=0;
    private SensorManager sensorManager;
    Sensor accelerometer;
    long starttime = 0;
    double[] mag = new double[50];
    int i=0;
    static double min,max,var,std;
    int interval=5; // 5 seconds

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        init();

                start_collect.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        code=0;
                        start_collect.setEnabled(false);
                        end_collect.setEnabled(true);
                        record=true;
                        Toast.makeText(ActivityAct.this, "saving data to activity.csv...", Toast.LENGTH_SHORT).show();
                    }
                });
                end_collect.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        code=1;
                        start_collect.setEnabled(true);
                        end_collect.setEnabled(false);
                        record=false;
                        Toast.makeText(ActivityAct.this, "Data saved on activity.csv", Toast.LENGTH_SHORT).show();
                        Log.d("Nazim","file content:\n"+readFileAsString(filename));
                    }
                });
        //train
        train_b.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(ActivityAct.this, "Training 3 models on activity.csv, loading evaluation...", Toast.LENGTH_SHORT).show();
                TrainWEKA();
            }
        });
        //deploy
        deploy_b.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //deploy activity
                code=2;
                Toast.makeText(ActivityAct.this, "Saved models Deployed on activity.csv", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void init(){
        setContentView(R.layout.activity_act);
        sensorManager = (SensorManager)  getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorManager.registerListener(ActivityAct.this,accelerometer,sensorManager.SENSOR_DELAY_NORMAL);
        //collect
        start_collect=(Button)findViewById(R.id.start_id);
        end_collect=(Button)findViewById(R.id.end_id);
        end_collect.setEnabled(false);
        edit_label=(EditText)findViewById(R.id.label_edit);

        //train
        train_text[0]=(TextView)findViewById(R.id.train_RF_id);
        train_text[1]=(TextView)findViewById(R.id.train_NB_id);
        train_text[2]=(TextView)findViewById(R.id.train_DT_id);
        train_b=(Button)findViewById(R.id.train_id);

        deploy_b=(Button)findViewById(R.id.deploy_id);
        deploy_text[0]=(TextView)findViewById(R.id.deploy_RF_id);
        deploy_text[1]=(TextView)findViewById(R.id.deploy_NB_id);
        deploy_text[2]=(TextView)findViewById(R.id.deploy_DT_id);
        code=1;

    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent){

        double magnitude = Math.sqrt(sensorEvent.values[0]*sensorEvent.values[0]+sensorEvent.values[1]*sensorEvent.values[1]+sensorEvent.values[2]*sensorEvent.values[2]);
        long millis = System.currentTimeMillis() - starttime;
        int seconds = (int) (millis / 1000);
        seconds     = seconds % 60;


        if (seconds % interval ==0 || i==50){
            min=minimum(mag);
            max=maximum(mag);
            var=variance(mag);
            std=standardDeviation(mag);
            if(code==2) {
                multimodalDeploy();
            }else if(code==0){
                //Collect
                if(record) {
                    try {
                        String label = edit_label.getText().toString();
                        String append_data = "" + seconds + "," + min + "," + max + "," + std + "," + var + "," + label+"\n";
                        //writeStringAsFile(append_data, filename);
                        writeFile(append_data,filename);
                        Log.d("Nazim","data:"+append_data);
                    }catch (Exception e){
                        Toast.makeText(ActivityAct.this,"Wrong input format",Toast.LENGTH_LONG).show();
                        Log.d("Nazim","error: "+e.toString());
                    }
                }
            }
            i=0;
            //Arrays.fill(mag, 0.0);
        } else{
            mag[i]=magnitude;
            i++;
        }
    }

    public void multimodalDeploy(){

            Thread t =new Thread(){
            @Override
            public void run() {
                while (!isInterrupted()) {
                    try {
                        Thread.sleep(5000);  //1000ms = 1 sec
            runOnUiThread(new Runnable() {
            @Override
            public void run() {
            double[] detected;
            float convert_acc;

            if(j==3)j=0;
            detected=classification(min, max, var, std, j);
            convert_acc = (float) Math.round(detected[1] * 100) / 100;

            if (detected[0] == 0) deploy_text[j].setText("walking: "+convert_acc);
            else if (detected[0] == 1) deploy_text[j].setText("sitting: "+convert_acc);
            else if (detected[0] == 2) deploy_text[j].setText("standing: "+convert_acc);
            else deploy_text[j].setText("laying: "+convert_acc);
            j++;
                }
            }); } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                } }
            };
        t.start();
    }

    public double [] classification(double min, double max, double var, double std,byte model){

        double [] predicted_class ={0,0};
        try {
                ArrayList<Attribute> attributes = new ArrayList<>();
                attributes.add(new Attribute("min", 0));
                attributes.add(new Attribute("max", 1));
                attributes.add(new Attribute("var", 2));
                attributes.add(new Attribute("std", 3));

                attributes.add(new Attribute("label", Arrays.asList("0", "1", "2", "3"), 4));
                // new instance to classify. Class label should be "no"
                Instance instance = new SparseInstance(4);
                instance.setValue(attributes.get(0), min);
                instance.setValue(attributes.get(1), max);
                instance.setValue(attributes.get(2), var);
                instance.setValue(attributes.get(3), std);
                // Create an empty set
                Instances datasetConfiguration;
                datasetConfiguration = new Instances("cervix.symbolic", attributes, 0);
                datasetConfiguration.setClassIndex(4);
                instance.setDataset(datasetConfiguration);

                double[] distribution={0,0};

                if(model==0){
                    distribution = tree.distributionForInstance(instance);
                    predicted_class[0] = tree.classifyInstance(instance);
                }else if(model==1){
                    distribution = nb.distributionForInstance(instance);
                    predicted_class[0] = nb.classifyInstance(instance);
                }else if(model==2){
                    distribution = lmt.distributionForInstance(instance);
                    predicted_class[0] = lmt.classifyInstance(instance);
                }else distribution=null;

                predicted_class[1] = Math.max(distribution[0], distribution[1]) * 100;
                Log.d("Nazim", "Results:" + predicted_class[0] + "/" + predicted_class[1]);

        } catch (Exception e) {
            e.printStackTrace();
        }
    return predicted_class;
    }

    public void TrainWEKA() {
        try{
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("/data/data/com.example.cancerpredictor/files/Nazim_activity.csv"));
            Instances data = loader.getDataSet();
            Log.d("Nazim","dataset size:"+data.numInstances());

            //preprocessing
            if(data.classIndex()==-1)data.setClassIndex(data.numAttributes()-1);
            NumericToNominal numericToNominal=new NumericToNominal();
            numericToNominal.setInputFormat(data);
            Instances train=Filter.useFilter(data,numericToNominal);

            tree=(Classifier)Class.forName("weka.classifiers.lazy.IBk").newInstance();
            tree.buildClassifier(train);
            Evaluation tree_eval=new Evaluation(train);
            tree_eval.crossValidateModel(tree,train,5, new Random(1));
            float convert_acc = (float) Math.round((1-tree_eval.errorRate()) * 100) / 100;
            train_text[0].setText("IBk:"+convert_acc);
            // serialize models and save trained models
            //saveModels(tree,"nazim_j48.model");

            nb=(Classifier)Class.forName("weka.classifiers.bayes.NaiveBayes").newInstance();
            nb.buildClassifier(train);
            Evaluation nb_eval=new Evaluation(train);
            nb_eval.crossValidateModel(nb,train,5, new Random(1));
            convert_acc = (float) Math.round((1-nb_eval.errorRate()) * 100) / 100;
            train_text[1].setText("NB:"+convert_acc);
            //saveModels(nb,"nazim_nb.model");

            /* PLZ UNCOMMENT THIS AND TEST WITH A BETTER PHONE*/
            lmt=(Classifier)Class.forName("weka.classifiers.lazy.LWL").newInstance();
            lmt.buildClassifier(train);
            Evaluation lmt_eval=new Evaluation(train);
            lmt_eval.crossValidateModel(lmt,train,5, new Random(1));
            convert_acc = (float) Math.round((1-lmt_eval.errorRate()) * 100) / 100;
            train_text[2].setText("LWL:"+convert_acc);
            //saveModels(lmt,"nazim_lmt.model");

        }catch (NumberFormatException e){
            e.printStackTrace();
            Toast.makeText(ActivityAct.this, e.toString(), Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void saveModels(Classifier classifier,String name) {

        try {
            ObjectOutputStream oos = new ObjectOutputStream(
                    new FileOutputStream(getAssets()+name,true));
            oos.writeObject(classifier);
            oos.flush();
            oos.close();
        }catch (Exception e){
            Log.d("Nazim",""+e.toString());
        }
        finally {
            Toast.makeText(ActivityAct.this, "model saved to assets folder", Toast.LENGTH_SHORT).show();
        }
    }

    public void writeFile(final String fileContents, String fileName)
    {
        Context context = getApplicationContext();
        try {
            FileWriter out = new FileWriter(new File(context.getFilesDir(), fileName),true);
            out.write(fileContents); out.close();
        } catch (IOException e)
        {
            Log.d("Nazim", e.toString());
        }
    }

    public String readFileAsString(String fileName)
    {
        Context context = getApplicationContext();
        StringBuilder stringBuilder = new StringBuilder();
        String line; BufferedReader in = null;
        try {
            in = new BufferedReader(new FileReader(new File(context.getFilesDir(),
                    fileName)));
            while ((line = in.readLine()) != null){
                stringBuilder.append(line+"\n");
                //visualise all attributes
                String[] tokens = line.split(",");
            }
        } catch (FileNotFoundException e) {
            Log.d("Nazim", e.toString());
        } catch (IOException e) {
            Log.d("Nazim", e.toString());
        }
        return stringBuilder.toString();
    }

    @Override
    public void  onAccuracyChanged(Sensor sensor, int i){
    }

    public static double minimum(double data[]){
        if(data == null || data.length == 0) return 0.0;
        int length = data.length;
        double MIN = data[0];
        for (int i = 1; i < length; i++){
            MIN = data[i]<MIN?data[i]:MIN;
        }
        return MIN;
    }

    public static double maximum(double data[]){
        if(data == null || data.length == 0) return 0.0;

        int length = data.length;
        double Max = data[0];
        for (int i = 1; i<length; i++)
            Max = data[i]<Max ? Max : data[i];
        return Max;
    }
    public static double variance(double data[]){
        if(data == null || data.length == 0) return 0.0;
        int length = data.length;
        double average = 0, s = 0, sum = 0;
        for (int i = 0; i<length; i++)
        {
            sum = sum + data[i];
        }
        average = sum / length;
        for (int i = 0; i<length; i++)
        {
            s = s + Math.pow(data[i] - average, 2);
        }
        s = s / length;
        return s;
    }
    public static double standardDeviation(double data[]){
        if(data == null || data.length == 0) return 0.0;
        double s = variance(data);
        s = Math.sqrt(s);
        return s;
    }
}
