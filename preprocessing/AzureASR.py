import json
import azure.cognitiveservices.speech as speechsdk


class AzureASR:
    def __init__(self, key_path='./keys/Azure-ASR-keys.json'):
        with open(key_path) as key_file:
            keys = json.load(key_file)
        speech_key, service_region = keys['key1'], keys['location']

        self.speech_config = speechsdk.SpeechConfig(
            subscription=speech_key, region=service_region)

    def recognize(self, file_path):
        audio_config = speechsdk.AudioConfig(filename=file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=audio_config)

        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            raise Exception("No speech could be recognized: {}".format(
                result.no_match_details))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            raise Exception("Speech recognition cancelled: {}".format(
                cancellation_details.reason))
        else:
            raise Exception("Unexpected recognition result: {}".format(result.reason))
