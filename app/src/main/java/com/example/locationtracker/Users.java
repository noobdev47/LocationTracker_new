package com.example.locationtracker;

public class Users {

    public String name,email,activity;

    public Users(){

    }

    public Users(String name, String email, String activity) {
        this.name = name;
        this.email = email;
        this.activity = activity;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getActivity() {
        return activity;
    }

    public void setActivity(String activity) {
        this.activity = activity;
    }
}
