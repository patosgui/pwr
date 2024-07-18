#!/usr/bin/env python3

import streamlit_authenticator as stauth
import sys

print("Hashing " + sys.argv[1])
print(stauth.utilities.hasher.Hasher([sys.argv[1]]).generate()[0])
