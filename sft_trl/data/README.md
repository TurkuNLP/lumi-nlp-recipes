## Examples on how to format data into chatml.

### Example conversation


```json
{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."}
  ]
}

{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"},
    {"role": "assistant", "content": "William Shakespeare is the author of 'Romeo and Juliet'."}
  ]
}

{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "How far is the Moon from Earth?"},
    {"role": "assistant", "content": "The average distance from Earth to the Moon is approximately 384,400 kilometers (238,855 miles)."}
  ]
}
