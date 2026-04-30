.users
| map(select(.active))
| map({id, name, email})
