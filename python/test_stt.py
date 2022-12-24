
# [START speech_transcribe_sync]
def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    import io
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r"C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/sesac-371212-227f22f8a69a.json"

    client = speech.SpeechClient()

    # [START speech_python_migration_sync_request]
    # [START speech_python_migration_config]
    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ko-KR",
    )
    # [END speech_python_migration_config]

    # [START speech_python_migration_sync_response]
    response = client.recognize(config=config, audio=audio)

    # [END speech_python_migration_sync_request]
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
    # [END speech_python_migration_sync_response]


# [END speech_transcribe_sync]


if __name__ == "__main__":
    transcribe_file('C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/audio.wav')