from django.contrib import admin
from .models import UserDetails, UserActivity, GuestUserActivity


# Register your models here.

admin.site.register(UserDetails)
admin.site.register(UserActivity)
admin.site.register(GuestUserActivity)
