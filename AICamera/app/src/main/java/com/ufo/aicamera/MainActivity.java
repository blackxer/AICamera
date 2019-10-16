package com.ufo.aicamera;

import java.io.File;
import java.io.FileNotFoundException;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.Window;
import android.widget.TextView;
import android.widget.ImageView;
import android.widget.Toast;
import android.widget.Button;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.content.ContentResolver;
import android.view.View.OnClickListener;
import android.net.Uri;
import android.media.MediaMetadataRetriever;

import java.nio.ByteBuffer;
import java.util.Arrays;

import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "AICamera";

    private Button selectImage;
    private Button selectVideo;
    private Button predict;
    private ImageView imageView;
    private ImageView result;
    private TextView tv;
    private String predictedClass = "none";
    private AssetManager mgr;
    private Bitmap srcBitmap;

    static {
        System.loadLibrary("native-lib");
    }

    public native String predFromCaffe2(Object bitmap);

    public native void initCaffe2(AssetManager mgr);

    private class SetUpNeuralNetwork extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void[] v) {
            try {
                initCaffe2(mgr);
                predictedClass = "Neural net loaded! Inferring...";
            } catch (Exception e) {
                Log.d(TAG, "Couldn't load neural network.");
            }
            return null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);

        mgr = getResources().getAssets();

        new SetUpNeuralNetwork().execute();

        View decorView = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
        decorView.setSystemUiVisibility(uiOptions);

        setContentView(R.layout.activity_main);

        tv = (TextView) findViewById(R.id.sample_text);

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }

        selectImage = (Button) this.findViewById(R.id.selectImage);
        selectVideo = (Button) this.findViewById(R.id.selectVideo);
        predict = (Button) this.findViewById(R.id.predict);
        imageView = (ImageView) this.findViewById(R.id.imageView);
        result = (ImageView) this.findViewById(R.id.result);

        selectImage.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // TODO 自动生成的方法存根

                Intent intent = new Intent();
                //打开pictures画面Type设置为image
                intent.setType("image/*");
                //使用Intent.ACTION_GET_CONTENT 这个Action
                intent.setAction(Intent.ACTION_GET_CONTENT);
                //取得像片后返回本画面
                startActivityForResult(intent, 2);
            }
        });
        predict.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                predictedClass = predFromCaffe2((Object)srcBitmap);
                tv.setText(predictedClass);
                imageView.setImageBitmap(srcBitmap);
                Toast.makeText(MainActivity.this, "predict done", Toast.LENGTH_LONG).show();
            }
        });

        selectVideo.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                new AsyncTask<Void, Bitmap, Void>(){
                    @Override
                    protected Void doInBackground(Void... voids) {
                        MediaMetadataRetriever mmr = new MediaMetadataRetriever();//实例化MediaMetadataRetriever对象
                        File file = new File("/sdcard/demo/1.mp4");//实例化File对象，文件路径为/storage/sdcard/Movies/music1.mp4
                        if(file.exists()){
                            mmr.setDataSource(file.getAbsolutePath());//设置数据源为该文件对象指定的绝对路径
                            // 取得视频的长度(单位为毫秒)
                            String time = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
                            // 取得视频的长度(单位为毫秒)
                            long melliseconds = Long.valueOf(time);

                            for (long i = 0; i < melliseconds; ) {
                                Bitmap bitmap = mmr.getFrameAtTime(i*1000, MediaMetadataRetriever.OPTION_CLOSEST);
                                publishProgress(bitmap);
                                i+=30;
                            }
                        }else{
                            Toast.makeText(MainActivity.this, "文件不存在", Toast.LENGTH_SHORT).show();//文件不存在时，弹出消息提示框
                        }
                        return null;
                    }
                    protected void onProgressUpdate(Bitmap... bitmap){
                        predictedClass = predFromCaffe2((Object) bitmap[0]);
                        result.setImageBitmap(bitmap[0]);//设置ImageView显示的图片
                        tv.setText(predictedClass);
                    }

                }.execute();

            }
        });

    }

    /**
     * 定义方法onActivityResult来处理用户挑选的图片。通过requestCode和resultCode返回标识码，数据类型为Intent的data参数，
     * 调用Intent对象的getData（）方法可以获得具体内容。
     */
    protected void onActivityResult(int requestCode,int resultCode,Intent data){
        if (resultCode==RESULT_OK){
            Uri uri=data.getData();
            ContentResolver cr=this.getContentResolver();
            try{
                srcBitmap = BitmapFactory.decodeStream(cr.openInputStream(uri));
                //将Bitmap设置到imageView
                imageView.setImageBitmap(srcBitmap);
            }catch(FileNotFoundException e)
            {
                e.printStackTrace();
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == 1) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(MainActivity.this, "You can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }


}
