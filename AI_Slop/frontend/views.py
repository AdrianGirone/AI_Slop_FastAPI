from django.shortcuts import render
from django.http import JsonResponse
from AI_Slop.ollama_client import query_ollama
from django.views.decorators.csrf import csrf_exempt


def index(request):
    """Render the main page with the text input form."""
    return render(request, 'index.html')


@csrf_exempt
def submit_text(request):
    """
    Handle text submission from the form and send it to Ollama.
    """
    if request.method == 'POST':
        text = request.POST.get('text', '')
        
        if not text:
            return JsonResponse({
                'status': 'error',
                'message': 'No text provided'
            }, status=400)
        
        try:
            # Send the text to Ollama as a prompt
            result = query_ollama("llama3", text)
            return JsonResponse({
                'status': 'success',
                'message': 'Response received from AI',
                'prompt': text,
                'response': result
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error: {str(e)}',
                'hint': 'Make sure Ollama is running (ollama serve) and the model is installed (ollama pull llama3)'
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def ai_response(request):
    user_prompt = request.GET.get("prompt", "Say hello!")  # simple GET param example
    result = query_ollama("llama3", user_prompt)
    return JsonResponse({"response": result})