.events
| map(select(.level == "error"))
| map({time, service, message})
